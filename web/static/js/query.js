document.addEventListener('DOMContentLoaded', function(){
     document.getElementById("query").addEventListener("click", function() {
            call_diva();
        }, false);

    function array(num){
        document.getElementById('videodisplay').currentTime = num;
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
            console.log(computeReturn);
            arrimg = computeReturn['file'].split(',');
            console.log(arrimg);
            document.getElementById('work').style.display = "block";
            document.getElementById('workimage').style.display = "block";
            document.getElementById('demo_tab').style.display = "block";
            if (document.getElementById('results').childNodes.length > 0) {
                var child = document.getElementById('results').lastElementChild;
                while (child) {
                    document.getElementById('results').removeChild(child);
                    child = document.getElementById('results').lastElementChild;
                }
            }
            for (let i = 1; i < arrimg.length; i+=2) {
                var img = new Image();
                //dir = arrimg[0] +'/' + arrimg[i];
                //img.src = dir;
                img.src = "{{ url_for('download_file', filename="+ arrimg[i]+") }}";
                img.width = 400;
                img.height = 300;
                img.addEventListener('click', function(){array(parseInt(arrimg[i+1]))}, false);
                // img.addEventListener('click', function(val){return function(){array(val);};}(i), false);
                document.getElementById('results').appendChild(img);
            }

        });
    }


},false);