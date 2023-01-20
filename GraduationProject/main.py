#!/usr/bin/env python

"""
Welcome to CARLA Simulator

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    A/D          : steer left/right
    Q            : toggle reverse
    Space        : hand-brake
    U            : increase top speed 
    J            : dicrease top speed 

    L            : Lane Departure System

    TAB          : change sensor positions
    C            : change weather (Shift+C reverse)

    F1           : toggle HUD
    H/?          : toggle help
    ESC          : quit
"""

from __future__ import print_function


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys

try:
    sys.path.append(glob.glob('../CARLA/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    raise RuntimeError(
        'cannot import carla, make sure carla package is installed')


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla

from carla import ColorConverter as cc

import argparse
import collections
import datetime
import logging
import math
import random
import weakref
import copy
import cantools
import can
import warnings

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_u
    from pygame.locals import K_j
    from pygame.locals import K_q
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_l

    from pathlib import Path
    from lane_detection.lane_detector import LaneDetector
    from lane_detection.camera_geometry import CameraGeometry
    from control.pure_pursuit import PurePursuitPlusPID
    from util.carla_util import carla_vec_to_np_array, carla_img_to_array, CarlaSyncMode, find_weather_presets
    from util.geometry_util import dist_point_linestring
except ImportError:
    raise RuntimeError(
        'cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================
fastai_model_paths=["lane_detection/fastai_model.pth","lane_detection/fastai_model_weather.pth"]

def get_trajectory_from_lane_detector(world, image):
    # Lane Detector
    ld=world.ld
    # get lane boundaries using the lane detector
    img = carla_img_to_array(image)
    poly_left, poly_right, left_mask, right_mask = ld.fit_polynomial(img)
    # If polynomials are poorly fitted, fits the polynomials with location thats for improvement
    # if(poly_left[0]==0 or poly_right[0]==0):
    #     return world.get_polylines()
    # trajectory to follow is the mean of left and right lane boundary
    x = np.arange(-2, 60, 1.0)
    y = -0.5*(poly_left(x)+poly_right(x))
    # x,y is now in coordinates centered at camera, but camera is 0.5 in front of vehicle center
    x += 0.5
    traj = np.stack((x, y)).T
    return traj
def ld_detection_overlay(image, left_mask, right_mask):
    res = copy.copy(image)
    res[left_mask > 0.5, :] = [0, 0, 255]
    res[right_mask > 0.5, :] = [255, 0, 0]
    return res


def send_control(vehicle, throttle, steer, brake,
                 hand_brake=False, reverse=False):
    throttle = np.clip(throttle, 0.0, 1.0)
    steer = np.clip(steer, -1.0, 1.0)
    brake = np.clip(brake, 0.0, 1.0)
    control = carla.VehicleControl(throttle, steer, brake, hand_brake, reverse)
    vehicle.apply_control(control)




def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


class World(object):
    def __init__(self, carla_world, hud, args):
        self.world = carla_world
        self.actor_role_name = args.rolename
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print(
                '  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.hud = hud
        self.player = None
        self.camera_rgb=None
        self.camera_windshield=None
        self.sensorss=[]
        self.cg=None
        self.ld=None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._gamma = args.gamma
        self.spawn_index = args.spawn
        self.model_index=args.model
        self.restart()
        self.world.on_tick(hud.on_world_tick)
        self.cross_track_error=0
        self.max_track_error=0
        self.recording_enabled = False
        self.recording_start = 0
        self.constant_velocity_enabled = False
        self.lane_departure_enabled = False
        self.desired_speed=20   #m/s

    def restart(self):
        blueprint_library = self.world.get_blueprint_library()
        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713
        m = self.world.get_map()
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Get a random blueprint.
        blueprint = random.choice(
            self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute('role_name', self.actor_role_name)
        if blueprint.has_attribute('color'):
            color = random.choice(
                blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(
                blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'true')
        
        while self.player is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()
            spawn_point = random.choice(
                spawn_points) if spawn_points else carla.Transform()

            veh_bp = random.choice(blueprint_library.filter('vehicle.tesla.model3'))
            veh_bp.set_attribute('color', '64,81,181')
            self.player = self.world.try_spawn_actor(veh_bp, m.get_spawn_points()[self.spawn_index])
            # self.player = self.world.try_spawn_actor(veh_bp, spawn_point) spawns randomly

        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        self.camera_rgb=self.world.spawn_actor(
            blueprint_library.find('sensor.camera.rgb'),
            carla.Transform(carla.Location(x=-5.5, z=2.8),
                            carla.Rotation(pitch=-10)),
            attach_to=self.player)
        # cg = CameraGeometry()
        cg = CameraGeometry(image_width=512, image_height=256)
        self.ld = LaneDetector(model_path=Path(
                    fastai_model_paths[self.model_index]).absolute(), cam_geom=cg)
        cam_windshield_transform = carla.Transform(carla.Location(
                x=0.5, z=cg.height), carla.Rotation(pitch=cg.pitch_deg))
        bp = blueprint_library.find('sensor.camera.rgb')
        fov = cg.field_of_view_deg
        bp.set_attribute('image_size_x', str(cg.image_width))
        bp.set_attribute('image_size_y', str(cg.image_height))
        bp.set_attribute('fov', str(fov))
        self.camera_windshield = self.world.spawn_actor(
            bp,
            cam_windshield_transform,
            attach_to=self.player)
        actor_type = get_actor_display_name(self.player)
        self.sensorss.append(self.camera_rgb)
        self.sensorss.append(self.camera_windshield)
        self.hud.notification(actor_type)

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    

    def tick(self, clock):
        self.hud.tick(self, clock)
    def get_polylines(self):
        wp = self.world.get_map().get_waypoint(self.player.get_transform().location)
        wps = [wp]
        for _ in range(20):
            next_wps = wp.next(1.0)
            if len(next_wps) > 0:
                wp = sorted(next_wps, key=lambda x: x.id)[0]
            wps.append(wp)

        # transform waypoints to vehicle ref frame
        poly_lines = np.array(
            [np.array([*carla_vec_to_np_array(x.transform.location), 1.])
            for x in wps]
        ).T
        trafo_matrix_world_to_vehicle = np.array(
            self.player.get_transform().get_inverse_matrix())

        poly_lines = trafo_matrix_world_to_vehicle @ poly_lines
        poly_lines = poly_lines.T
        poly_lines = poly_lines[:, :2]
        return poly_lines   
    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        sensors = [
            self.camera_manager.sensor,
            self.camera_windshield,
            self.camera_rgb]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        if self.player is not None:
            self.player.destroy()
    

# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    """Class that handles keyboard input."""

    def __init__(self, world):
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            self._lights = carla.VehicleLightState.NONE
            world.player.set_light_state(self._lights)
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self, client, world, clock):
        if isinstance(self._control, carla.VehicleControl):
            current_lights = self._lights
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_u:
                    world.desired_speed+=2
                elif event.key == K_j:
                    if(world.desired_speed != 10):
                        world.desired_speed-=2
                elif event.key == K_l:
                    world.lane_departure_enabled = not world.lane_departure_enabled
                if event.key == K_q:
                        self._control.gear = 1 if self._control.reverse else -1  

        
        if isinstance(self._control, carla.VehicleControl):
            self._parse_vehicle_keys(world,
                pygame.key.get_pressed(), clock.get_time())
            self._control.reverse = self._control.gear < 0
            # Set automatic control-related vehicle lights
            if self._control.brake:
                current_lights |= carla.VehicleLightState.Brake
            else:  # Remove the Brake flag
                current_lights &= ~carla.VehicleLightState.Brake
            if self._control.reverse:
                current_lights |= carla.VehicleLightState.Reverse
            else:  # Remove the Reverse flag
                current_lights &= ~carla.VehicleLightState.Reverse
            if current_lights != self._lights:  # Change the light state only if necessary
                self._lights = current_lights
                world.player.set_light_state(
                    carla.VehicleLightState(self._lights))
        world.player.apply_control(self._control)

    def _parse_vehicle_keys(self,world, keys, milliseconds):
        if keys[K_UP] or keys[K_w]:
            self._control.throttle = min(self._control.throttle + 0.1, 1)
        else:
            self._control.throttle = 0.0

        if keys[K_DOWN] or keys[K_s]:
            self._control.brake = min(self._control.brake + 0.2, 1)
        else:
            self._control.brake = 0
            
        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            if self._steer_cache > 0:
                self._steer_cache = 0
            else:
                self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            if self._steer_cache < 0:
                self._steer_cache = 0
            else:
                self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.hand_brake = keys[K_SPACE]

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)

# ==============================================================================
# -- CAN -------------------------------------------------------------- ----- --
# ==============================================================================


class CAN(object):
    def __init__(self):
        self.db = cantools.database.load_file('control/car.dbc')
        self.can_bus = can.interface.Bus('vcan0', bustype='socketcan')
        self.speed_message = self.db.get_message_by_name('WHEEL_SPEEDS')
        self.steer_message = self.db.get_message_by_name('STEERING_SENSORS')
        self.speed_cache = 0
        self.steer_cache = 0
    def send_car_speed(self, speed):
        if int(speed) != int(self.speed_cache):
            self.speed_cache = speed
            data = self.speed_message.encode({'WHEEL_SPEED_FL': speed})
            message = can.Message(arbitration_id=self.speed_message.frame_id, data=data)
            self.can_bus.send(message)


    def send_steering(self, steer):
        if steer  != self.steer_cache:
            self.steer_cache = steer
            data = self.steer_message.encode({'STEER_ANGLE': steer * 500, 'STEER_WHEEL_ANGLE': steer * 500})
            message = can.Message(arbitration_id=self.steer_message.frame_id, data=data)
            self.can_bus.send(message)
        

    
# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 16), width, height)
        self.server_fps = 0
        self.frame = 0
        self.can = CAN()
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        
        vehicles = world.world.get_actors().filter('vehicle.*')
        self.can.send_car_speed(int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)))
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(
                world.player, truncate=20),
            'Map:     % 20s' % world.map.name,
            'Simulation time: % 12s' % datetime.timedelta(
                seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 *
                                       math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            'Desired Speed:% 2.0f km/h' % (world.desired_speed * 3.6),
            "lateral error (cm):  {}".format(world.cross_track_error),
            # "max lat. error (cm): {}".format(world.max_track_error),
            'Location:% 20s' % ('(% 5.1f, % 5.1f, % 5.1f)' %
                                (t.location.x, t.location.y,t.location.z)),
            '']
        if isinstance(c, carla.VehicleControl):
            self.can.send_steering(c.steer)
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear),
                'Lane Departure System: %s' % {True: 'ON', False: 'OFF'}.get(world.lane_departure_enabled, world.lane_departure_enabled)]

        
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            def distance(l): return math.sqrt((l.x - t.location.x) **
                                              2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x)
                        for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles, key=lambda vehicles: vehicles[0]):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30)
                                  for x, y in enumerate(item)]
                        pygame.draw.lines(
                            display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect(
                            (bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255),
                                         rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect(
                            (bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(
                            display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect(
                                (bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect(
                                (bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(
                        item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """Helper class to handle text output using pygame"""

    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.line_space = 18
        self.dim = (780, len(lines) * self.line_space + 12)
        self.pos = (0.5 * width - 0.5 *
                    self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * self.line_space))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)



# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud, gamma_correction):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        Attachment = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(carla.Location(x=-5.5, z=2.5),
             carla.Rotation(pitch=8.0)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid),
            (carla.Transform(carla.Location(x=5.5, y=1.5, z=1.5)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=-8.0, z=6.0),
             carla.Rotation(pitch=6.0)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=-1, y=-bound_y, z=0.5)), Attachment.Rigid)]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)', {}],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)', {}],
            ['sensor.camera.depth', cc.LogarithmicDepth,
                'Camera Depth (Logarithmic Gray Scale)', {}],
            ['sensor.camera.semantic_segmentation', cc.Raw,
                'Camera Semantic Segmentation (Raw)', {}],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
                'Camera Semantic Segmentation (CityScapes Palette)', {}],
            ['sensor.lidar.ray_cast', None,
                'Lidar (Ray-Cast)', {'range': '50'}],
            ['sensor.camera.dvs', cc.Raw, 'Dynamic Vision Sensor', {}],
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB Distorted',
                {'lens_circle_multiplier': '3.0',
                 'lens_circle_falloff': '3.0',
                 'chromatic_aberration_intensity': '0.5',
                 'chromatic_aberration_offset': '0'}]]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(gamma_correction))
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
            elif item[0].startswith('sensor.lidar'):
                self.lidar_range = 50

                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
                    if attr_name == 'range':
                        self.lidar_range = float(attr_value)

            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index +
                                1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else \
            (force_respawn or (self.sensors[index]
             [2] != self.sensors[self.index][2]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(
                lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index


    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / (2.0 * self.lidar_range)
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        elif self.sensors[self.index][0].startswith('sensor.camera.dvs'):
            # Example of converting the raw_data from a carla.DVSEventArray
            # sensor into a NumPy array and using it as an image
            dvs_events = np.frombuffer(image.raw_data, dtype=np.dtype([
                ('x', np.uint16), ('y', np.uint16), ('t', np.int64), ('pol', np.bool)]))
            dvs_img = np.zeros((image.height, image.width, 3), dtype=np.uint8)
            # Blue is positive, red is negative
            dvs_img[dvs_events[:]['y'], dvs_events[:]
                    ['x'], dvs_events[:]['pol'] * 2] = 255
            self.surface = pygame.surfarray.make_surface(
                dvs_img.swapaxes(0, 1))
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)

# When user drive manualy, lane departure is passive
def onLaneDeparture(keys):
    return not (keys[K_UP] or keys[K_w] or keys[K_DOWN] or keys[K_s] or keys[K_LEFT] or keys[K_a] or keys[K_RIGHT] or keys[K_d])
# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================

def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(5.0)
        client.load_world('Town04')
        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud, args)
        
        manual_controller = KeyboardControl(world)
        pure_pid_controller = PurePursuitPlusPID()
        clock = pygame.time.Clock()
        frame=0
        FPS=15
        flag=0
        crossedLane=0
        m = client.get_world().get_map()
        with CarlaSyncMode(world.world, *world.sensorss, fps=FPS) as sync_mode:
            while True:

                clock.tick()
                tick_response = sync_mode.tick(timeout=2.0)
                clock.tick_busy_loop(20)
                
                snapshot, image_rgb, image_windshield = tick_response
                traj=world.get_polylines()
                # if(frame%2==0):
                #     traj = get_trajectory_from_lane_detector(
                #         world, image_windshield)
                    
                speed = np.linalg.norm(
                    carla_vec_to_np_array(world.player.get_velocity()))
                
                throttle, steer = pure_pid_controller.get_control(
                    traj, speed, desired_speed=world.desired_speed, dt=1./FPS)
        
                if manual_controller.parse_events(client, world, clock):
                        return
                dist = dist_point_linestring(np.array([0, 0]), traj)
                # dist = dist_point_linestring(np.array([0, 0]), world.get_polylines()) #true aproach
                world.cross_track_error=int(dist*100)
                world.max_track_error=max(world.max_track_error, world.cross_track_error)
                if(world.cross_track_error>59):
                    flag +=1
                    if(flag==1): # only first track error
                        crossedLane= 'right' if steer>0 else 'left'
                        lane_message=can.Message(arbitration_id=100, data=crossedLane.encode())
                        hud.can.can_bus.send(lane_message)
                    # send_control(world.player, throttle, steer, 0)
                else:
                    flag=0
                    message = can.Message(arbitration_id=100, data=''.encode())
                    hud.can.can_bus.send(message)
                if(world.lane_departure_enabled and onLaneDeparture(pygame.key.get_pressed())):
                        send_control(world.player, throttle, steer, 0) 
                world.tick(clock)
                world.render(display)
                pygame.display.flip()
                frame += 1
    finally:

        if (world and world.recording_enabled):
            client.stop_recorder()

        if world is not None:
            world.destroy()

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1920x1080',
        help='window resolution (default: 1920x1080)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '--spawn',
        default=90,
        type=int,
        help='spawn_index for spawning vehicle on the map (default: 90)')
    argparser.add_argument(
        '--model',
        default=0,
        type=int,
        help='different conditions for lane detection model (default: 0)')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]
    warnings.simplefilter('ignore', np.RankWarning)
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
