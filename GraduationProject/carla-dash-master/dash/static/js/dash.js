$(function(){
    window.setInterval(function(){
        $.ajax({
            url: '/canbus',
            contentType: 'application/json',
            success: function(response){
                
                
                document.getElementById("time").innerHTML=response['time'];
                document.getElementById("speed").innerHTML=response['speed'];
                var steer = response['steering']
                var crossedLane= response['crossed_lane']
                if(crossedLane!=''){
                    document.getElementById(crossedLane).style.visibility="visible";
                }
                else{
                    document.getElementById("left").style.visibility="hidden";
                    document.getElementById("right").style.visibility="hidden";
                }
                document.getElementById('steering').style['transform']='rotate('+steer/1.5+'deg)'
                document.getElementById('vehicle_img').style.transform='rotate('+(steer/20)+'deg)'
                //user-vehicle
            }
        });
      }, 10);
});