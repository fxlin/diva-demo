// xzl: load/display videos, etc. 

document.addEventListener('DOMContentLoaded', function(){
    let video_name;
    let duration;
    let timestamp = 0;
    let camera_name;
    let camera_address;
    let offset = 0;
    let video_URL;
    let score_URL;
    let image_URL;
    let index = 0;
    let id = null;
    let elem = null;

    var slider = document.getElementById("myRange");
    var output = document.getElementById("demo");
    output.innerHTML = slider.value;

    slider.oninput = function() {
      output.innerHTML = this.value;
    };

     document.getElementById("video1").addEventListener("click", function() {
            video(0);
        }, false);

     document.getElementById("video2").addEventListener("click", function() {
            video(1);
        }, false);

     document.getElementById("video3").addEventListener("click", function() {
            video(2);
        }, false);

     document.getElementById("video4").addEventListener("click", function() {
            video(3);
        }, false);

    function video(a){
        vid = a;
        index = a;
        document.getElementById('slider_0').style.display = "block";
        document.getElementById('vidtitle').style.display = "block";
        document.getElementById('video').style.display = "block";
        document.getElementById('video_tab').style.display = "block";
        console.log(video_URL);
        slider.max = duration[index];
        console.log(duration[index]);
        if(document.getElementById('videodisplay').src === video_URL[a]){

        }
        else {
            document.getElementById('videodisplay').src = video_URL[a];
        }
    }


    document.getElementById("sort_time").addEventListener("click", function() {
            sort_time();
        }, false);


    document.getElementById("sort_confidence").addEventListener("click", function() {
            sort_confidence();
        }, false);

    document.getElementById("query").addEventListener("click", function() {
            call_diva_2();
        }, false);

    document.getElementById("vid_req").addEventListener("click", function() {
            request_videos();
            // slider.max = "100";
            // update_progress_bar(parseInt(slider.max));
            }, false);

    function array(num){
        document.getElementById('videodisplay').currentTime = num;
    }

    function sort_confidence(){
        let temp_array = [];
        let child = document.getElementById('results').children;
        console.log(child.length);
         for (let i = 0; i < child.length; i+=1){
              temp_array.push([child[i].src , child[i].getAttribute('time')
                  , child[i].getAttribute('confidence')])
         }
         temp_array = temp_array.sort(function(a, b){return b[2]-a[2]});
           // if (document.getElementById('results').childNodes.length > 0) {
            child = document.getElementById('results').lastElementChild;
            while (child) {
                document.getElementById('results').removeChild(child);
                child = document.getElementById('results').lastElementChild;
            }
         // }

         for (let i = 0; i < temp_array.length; i++){
                let img = new Image();
                img.src = temp_array[i][0];
                img.width = 400;
                img.height = 300;
                img.setAttribute("time", temp_array[i][1]);
                img.setAttribute("confidence", temp_array[i][2]);
                img.title = 'Confidence Score: ' + temp_array[i][2] + '\n Timestamp: ' + temp_array[i][1];
                // img.addEventListener('click', function () {
                //     array(temp_array[i][1]);
                // }, false);
                document.getElementById('results').appendChild(img);
        }
         console.log(temp_array);
    }


    function sort_time(){
        let temp_array = [];
        let child = document.getElementById('results').children;
         for (let i = 0; i < child.length; i+=1){
              temp_array.push([child[i].src , child[i].getAttribute('time')
                  , child[i].getAttribute('confidence')])
         }
         temp_array = temp_array.sort(function(a, b){return a[1]-b[1]});

         // if (document.getElementById('results').childNodes.length > 0) {
             child = document.getElementById('results').lastElementChild;
            while (child) {
                document.getElementById('results').removeChild(child);
                child = document.getElementById('results').lastElementChild;
            }
         // }

         for (let i = 0; i < temp_array.length; i++){
                let img = new Image();
                img.src = temp_array[i][0];
                img.width = 400;
                img.height = 300;
                img.setAttribute("time", temp_array[i][1]);
                img.setAttribute("confidence", temp_array[i][2]);
                img.title = 'Confidence Score: ' + temp_array[i][2] + '\n Timestamp: ' + temp_array[i][1];
                console.log(temp_array[i][1]);
                // img.addEventListener('click', function () {
                //     array(temp_array[i][1]);
                // }, false);
                document.getElementById('results').appendChild(img);
        }
         console.log(temp_array);
    }

    // xzl: get metadata for all stored videos
    function request_videos(){
        let computeReturn = $.ajax({
                method: "POST",
                contentType: 'application/json',
                url: "/request_video",
        }).done(function (computeReturn) {
            document.getElementById('vid_req').style.display = "none";
            document.getElementById('obj_dropdown').style.display = "block";
            document.getElementById('img_1').style.display = "block";
             document.getElementById('img_2').style.display = "block";
             document.getElementById('img_3').style.display = "block";
             document.getElementById('img_4').style.display = "block";
            console.log(computeReturn);
            video_name = computeReturn['video'];  //
            duration = computeReturn['frame'];  // duration of video in seconds. Videos being processed are 30 fps.
            video_URL = computeReturn['video_URL'];  //
            console.log(duration);
            camera_name = computeReturn['camera_name'];
            camera_address = computeReturn['camera_address'];
            console.log(typeof(duration), duration);
        });
    }

    function call_diva_2(){
        // console.log("good");
        document.getElementById('work').style.display = "block";
        document.getElementById('workimage').style.display = "block";
        document.getElementById('demo_tab').style.display = "block";
        console.log(duration[index]);
        // console.log(slider.max, typeof(slider.max));
        document.getElementById("query").disabled = true;
        update_progress_bar(parseInt(duration[index]) / 30);
        let computeReturn = $.ajax({
            method: "POST",
            contentType: 'application/json',
            dataType: "json",
            data:JSON.stringify({'object':obj, 'video': video_URL[index], 'camera_name': camera_name[index], 'timestamp': timestamp,
            'offset': slider.value, 'camera_address': camera_address[index]
            }),
            url: "/display",
         }).done(function (computeReturn) {
             image_URL = computeReturn['image_url'];
             score_URL = computeReturn['conf'];
             console.log(image_URL);
             console.log(score_URL);
             clearInterval(id);
             elem.textContent = "Processed: " + 100 + "%";
             elem.style.width = 100 + "%";
             if (document.getElementById('results').childNodes.length > 0) {
                let child = document.getElementById('results').lastElementChild;
                while (child) {
                    document.getElementById('results').removeChild(child);
                    child = document.getElementById('results').lastElementChild;
                }
             }
             display_images();
             document.getElementById("query").disabled = false;
        });
    }



    function display_images() {

        for (let i = 0; i < image_URL.length; i += 1) {
            // let img = new Image();
            let img = document.createElement('img');
            img.src = image_URL[i];

            img.width = 400;
            img.height = 300;
            let time = image_URL[i].split('/');
            time = time[time.length - 1];
            time = time.split('.');
            time = time[0];
            console.log(time);
            img.title = 'Confidence Score: ' + score_URL[time]['0'] + '\n Timestamp: ' + time;
            img.setAttribute("time", parseInt(time));
            img.setAttribute("confidence", score_URL[time]['0']);
            // img.addEventListener('click', function () {
            //     array(time);
            // }, false);
            document.getElementById('results').appendChild(img);
        }
    }

    function reset_progress_bar(){
        let elem = document.getElementById("myBar");
        elem.style.width =  "0%";
        elem.setAttribute("aria-valuenow", 0);
    }


    function update_progress_bar(total) {
        console.log(total, typeof total);
        reset_progress_bar();
        elem = document.getElementById("myBar");
        let width = 0;
        let proc = 0;
        id = setInterval(frame, 920); // xzl: call frame() periodically... bad
        let temp_index = 0;
        let temp = 0;
        function frame() {
            console.log(width);
            if (width >= 100 && proc === total && temp_index % 10 === 0) {
                console.log("done");
                clearInterval(id);
            }
            else {
                width = ((width * 0.01 * total)+1) / total * 100;
                if(width > 100){
                    width = 99;
                }
                console.log(width);
                elem.style.width = width + "%";
                proc = parseInt(total * width * 0.01, 10);
                temp += 1;
                console.log(proc, total);
                elem.textContent = Math.floor(width) + "%";
            }
        }
    }

},false);