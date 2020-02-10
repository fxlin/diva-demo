document.addEventListener('DOMContentLoaded', function(){

    let pollingTime = null;
    let vid_name = '';


    document.getElementById("query").addEventListener("click", function() {
            call_diva();
        }, false);


    function array(num){
        document.getElementById('videodisplay').currentTime = num;
    }


    function retrieve_video_name(){
        let i;
        let temp;
        let video_name = '';
        temp = vid_name.split('/');
        temp = temp[temp.length-1];
        temp = temp.split('.');
        for (i=0; i < temp.length - 1; i++){
            if(i > 0){
                video_name += '.';
            }
            video_name += temp[i];

        }
        return video_name;
    }


    function display_images() {
        console.log('done waiting');
        let computeReturn = $.ajax({
            method: "POST",
            contentType: 'application/json',
            dataType: "json",
            data: JSON.stringify({'video': vid}),
            url: "/retrieve",
        }).done(function (computeReturn) {
            //  Print statements for debugging
            console.log(computeReturn);

            let arrImg = computeReturn['file'].split(',');
            let time = computeReturn['t'].split(',');

            //  Print statements for debugging
            console.log(arrImg);
            console.log(time);

            document.getElementById('work').style.display = "block";
            document.getElementById('workimage').style.display = "block";
            document.getElementById('demo_tab').style.display = "block";
            if (document.getElementById('results').childNodes.length > 0) {
                let child = document.getElementById('results').lastElementChild;
                while (child) {
                    document.getElementById('results').removeChild(child);
                    child = document.getElementById('results').lastElementChild;
                }
            }
            for (let i = 0; i < arrImg.length; i += 1) {
                let img = new Image();
                img.src = './static/output/' + retrieve_video_name() + '/' + arrImg[i] + '.jpg';
                img.width = 400;
                img.height = 300;
                img.addEventListener('click', function () {
                    array(parseInt(time[i]))
                }, false);
                document.getElementById('results').appendChild(img);
            }
            if (computeReturn['status'] === false) {
                pollingTime = window.setTimeout(display_images, 10000);
            } else {
                window.clearTimeout(pollingTime);
                document.getElementById("query").disabled = false;
            }

        });
    }


    function call_diva() {
        vid_name = vid;

        //  Print statements for debugging
        console.log(vid);
        console.log(obj);

        document.getElementById("query").disabled = true;
        let computeReturn = $.ajax({
            method: "POST",
            contentType: 'application/json',
            dataType: "json",
            data: JSON.stringify({'video': vid, 'object':obj}),
            url: "/display",
         }).done(function (computeReturn) {
             if (document.getElementById('results').childNodes.length > 0) {
                let child = document.getElementById('results').lastElementChild;
                while (child) {
                    document.getElementById('results').removeChild(child);
                    child = document.getElementById('results').lastElementChild;
                }
             }
             console.log("going to call to retrieve images from database in 10 seconds");
             pollingTime = window.setTimeout(display_images, 10000);
        });
    }


},false);