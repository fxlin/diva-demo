document.addEventListener('DOMContentLoaded', function(){
    let pollingTime = null;
    document.getElementById("query").addEventListener("click", function() {
            call_diva();
        }, false);

    function array(num){
        document.getElementById('videodisplay').currentTime = num;
    }

    function display_images() {
        var computeReturn = $.ajax({
            method: "POST",
            contentType: 'application/json',
            dataType: "json",
            data: JSON.stringify({'video': vid}),
            url: "/retrieve",
         }).done(function (computeReturn) {
            console.log(computeReturn);
            if(computeReturn['file'] === false){
                pollingTime = window.setTimeout(display_images, 10000);
                return;
            }
            else{
                window.clearTimeout(pollingTime);
            }
            arrimg = computeReturn['file'].split(',');
            console.log(arrimg);
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
            for (let i = 1; i < arrimg.length; i += 2) {
                let img = new Image();
                let dir = arrimg[0] +'/' + arrimg[i];
                img.src = dir;
                img.src = "{{ url_for('download_file', filename=" + arrimg[i] + ") }}";
                img.width = 400;
                img.height = 300;
                img.addEventListener('click', function () {
                    array(parseInt(arrimg[i + 1]))
                }, false);
                // img.addEventListener('click', function(val){return function(){array(val);};}(i), false);
                document.getElementById('results').appendChild(img);

            }
        });
    }

    function call_diva() {
        console.log(vid);
        console.log(obj);
        var computeReturn = $.ajax({
            method: "POST",
            contentType: 'application/json',
            dataType: "json",
            data: JSON.stringify({'video': vid, 'object':obj}),
            url: "/display",
         }).done(function (computeReturn) {
             display_images()
        });
    }


},false);