document.addEventListener('DOMContentLoaded', function(){
    let video_name;
    let obj_name;
    let duration;
    let timestamp = 0;
    let camera_name;
    let camera_address;
    let offset = 0;
    let video_URL;
    let score_URL;
    let image_URL;
    document.getElementById("vid_req").addEventListener("click", function() {
            request_videos();
            }, false);

    function request_videos(){
        let computeReturn = $.ajax({
                method: "POST",
                contentType: 'application/json',
                url: "/request_video",
        }).done(function (computeReturn) {
            console.log(computeReturn);
            video_name = computeReturn['video'];  //
            duration = computeReturn['frame'];  // duration of video in seconds. Videos being processed are 30 fps.
            video_URL = computeReturn['video_URL'];  //

            camera_name = computeReturn['camera_name'];
            camera_address = computeReturn['camera_address'];
            //
            call_diva_2();
        });
    }

    function call_diva_2(){
        console.log("good");
        let computeReturn = $.ajax({
            method: "POST",
            contentType: 'application/json',
            dataType: "json",
            data:JSON.stringify({'object':'motorbike', 'video': video_URL, 'camera_name': camera_name, 'timestamp': timestamp,
            'offset': offset, 'camera_address': camera_address
            }),
            url: "/display",
         }).done(function (computeReturn) {
             console.log(computeReturn);
             refresh();
             window.setTimeout(refresh,3000);
        });
    }


    function refresh(){
        console.log("refresh");
        let computeReturn = $.ajax({
            method: "POST",
            contentType: 'application/json',
            dataType: "json",
            data:JSON.stringify({ 'video': video_name, //
                'frame':duration,
                'video_URL':video_URL,  //
                'camera_name': camera_name,
                'camera_address':camera_address
            }),
            url: "/refresh",
         }).done(function (computeReturn) {


        });
    }


},false);