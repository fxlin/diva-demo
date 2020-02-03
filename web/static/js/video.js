document.addEventListener('DOMContentLoaded', function(){

     document.getElementById("video1").addEventListener("click", function() {
            video('/static/video/sonic.mp4');
        }, false);

     document.getElementById("video2").addEventListener("click", function() {
            video('/static/video/traffic_cam_vid.mp4');
        }, false);

     document.getElementById("video3").addEventListener("click", function() {
            video('/static/video/example.mp4');
        }, false);

    function video(a){
        vid = a;
        document.getElementById('vidtitle').style.display = "block";
        document.getElementById('video').style.display = "block";
        document.getElementById('video_tab').style.display = "block";
        if(document.getElementById('videodisplay').src === a){

        }
        else {
            document.getElementById('videodisplay').src = a;
        }
    }

},false);