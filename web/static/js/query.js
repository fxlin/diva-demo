document.addEventListener('DOMContentLoaded', function(){

    let pollingTime = null;
    let vid_name = '';

    document.getElementById("sort_time").addEventListener("click", function() {
            sort_time();
        }, false);


    document.getElementById("sort_confidence").addEventListener("click", function() {
            sort_confidence();
        }, false);

    document.getElementById("query").addEventListener("click", function() {
            call_diva();
        }, false);

    function array(num){
        console.log(num);
        document.getElementById('videodisplay').currentTime = num;
    }

    function duplicates(arrimg){
        let img_names = [];
        let temp = '';
        let ans = [];
        let child = document.getElementById('results').children;
        arrimg.sort(function(a, b){return a-b});
        console.log(arrimg);
         if(child.length === 0){
             return arrimg;
         }
         for (let i = 0; i < child.length; i+=1){
              temp = child[i].src.split('/');
              temp = temp[temp.length - 1];
              temp = temp.split('.')[0];
              img_names.push(temp);
         }
         for(let i = 0; i < arrimg.length; i += 1){
             if(img_names.indexOf(arrimg[i]) === -1) {
                 ans.push(arrimg[i]);
             }
         }
        return ans;
    }


    function sort_confidence(){
        let temp_array = [];
        let child = document.getElementById('results').children;
         for (let i = 0; i < child.length; i+=1){
              temp_array.push([child[i].src , child[i].getAttribute('time')
                  , child[i].getAttribute('confidence')])
         }
         temp_array = temp_array.sort(function(a, b){return a[2]-b[2]});
           if (document.getElementById('results').childNodes.length > 0) {
            let child = document.getElementById('results').lastElementChild;
            while (child) {
                document.getElementById('results').removeChild(child);
                child = document.getElementById('results').lastElementChild;
            }
         }

         for (let i = 0; i < temp_array.length; i++){
                let img = new Image();
                img.src = temp_array[i][0];
                img.width = 400;
                img.height = 300;
                img.setAttribute("time", temp_array[i][1]);
                img.setAttribute("confidence", temp_array[i][2]);
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

         if (document.getElementById('results').childNodes.length > 0) {
            let child = document.getElementById('results').lastElementChild;
            while (child) {
                document.getElementById('results').removeChild(child);
                child = document.getElementById('results').lastElementChild;
            }
         }

         for (let i = 0; i < temp_array.length; i++){
                let img = new Image();
                img.src = temp_array[i][0];
                img.width = 400;
                img.height = 300;
                img.setAttribute("time", temp_array[i][1]);
                img.setAttribute("confidence", temp_array[i][2]);
                document.getElementById('results').appendChild(img);
        }
         console.log(temp_array);
    }


    function test_duplicates(){
        let arrimg = ["0", "1", "2", '3', '10', '11', '12', '6', '7'];
        let i = 0;
        for (i in arrimg){
                let img = new Image();
                if(i === 0) {
                    img.src = './static/img/' + '11' + '.jpg';
                }
                else{
                    img.src = './static/img/' + '12' + '.jpg';
                }
                img.width = 400;
                img.height = 300;
                img.setAttribute("time", parseInt(i));
                img.setAttribute("confidence", -parseInt(i)/ 12);
                document.getElementById('results').appendChild(img);
        }
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
            let dict = {};
            for(let i = 0; i < arrImg.length; i++){
                dict[arrImg[i]] = time[i];
            }
            document.getElementById('work').style.display = "block";
            document.getElementById('workimage').style.display = "block";
            document.getElementById('demo_tab').style.display = "block";

            arrImg = duplicates(arrImg);

            for (let i = 0; i < arrImg.length; i += 1) {
                let img = new Image();
                img.src = './static/output/' + retrieve_video_name() + '/' + arrImg[i] + '.jpg';
                img.width = 400;
                img.height = 300;
                img.setAttribute("time", dict[arrImg[i]]);
                img.setAttribute("confidence", dict[arrImg[i]]);
                img.addEventListener('click', function () {
                    array(parseInt(dict[arrImg[i]]))
                }, false);
                document.getElementById('results').appendChild(img);
            }

            if (computeReturn['status'] === false) {
                pollingTime = window.setTimeout(display_images, 10000);
            }
            else {
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
             document.getElementById('work').style.display = "block";
             document.getElementById('workimage').style.display = "block";
             document.getElementById('demo_tab').style.display = "block";
             console.log("going to call to retrieve images from database in 10 seconds");
             pollingTime = window.setTimeout(display_images, 10000);
             update_progress_bar();
        });
    }
    //  These functions control the progress bar

    function reset_progress_bar(){
        let elem = document.getElementById("myBar");
        elem.style.width =  "0%";
        elem.setAttribute("aria-valuenow", 0);
    }


    function update_progress_bar() {
        reset_progress_bar();
        let elem = document.getElementById("myBar");
        let width = 0;
        let id = setInterval(frame, 1000);
        function frame() {
            let computeReturn = $.ajax({
                method: "POST",
                contentType: 'application/json',
                dataType: "json",
                data: JSON.stringify({'video': video_name}),
                url: "/ret_num",
             }).done(function (computeReturn) {
                if (width >= 100) {
                    console.log("done");
                    clearInterval(id);
                }
                else {
                    width = computeReturn['processed'] / computeReturn['total'] * 100;
                    elem.style.width = width + "%";
                    elem.textContent = "Processed: " + width + "%   (" + computeReturn['processed'] + "/"+ computeReturn['total'] + ")";
                }
            });
        }
    }



},false);