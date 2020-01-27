//  FIXME: need to extract coordinates from database
//         need to use these coordinates to draw bounding boxes against object
//         need to scale these coordinates based on the image size that is projected on the web page since its auto fitted

let bounding_box = {1.1:[100,200,30,40],2.2:[100,200,30,40], 5.5:[100,200,30,40], 10.6:[100,200,30,40],
    14.7:[100,200,30,40], 16.8:[100,200,30,40], 17.9:[100,200,30,40]
};
document.addEventListener('DOMContentLoaded', function(){
    var v = document.getElementById('videodisplay');
    var canvas = document.getElementById('c');
    var context = canvas.getContext('2d');
    var back = document.createElement('canvas');
    var backcontext = back.getContext('2d');
    var cw,ch;
    v.addEventListener('play', function(){
        cw = v.clientWidth;
        ch = v.clientHeight;
        canvas.width = cw;
        canvas.height = ch;
        back.width = cw;
        back.height = ch;
        draw(v,context,backcontext,cw,ch);
    },false);

},false);

function draw(v,c,bc,w,h) {

    // If video is paused or has ended then do not draw new canvas
    if(v.paused || v.ended) return false;
    // First, draw it into the backing canvas
    bc.drawImage(v,0,0,w,h);
    // Grab the pixel data from the backing canvas
    var idata = bc.getImageData(0,0,w,h);
    var data = idata.data;
    idata.data = data;
    // Draw the pixels onto the visible canvas
    let cur_time = v.currentTime.toFixed(1);
    if(cur_time in bounding_box){
        console.log(v.currentTime.toFixed(1));
        c.putImageData(idata,0,0);
        c.beginPath();
        let b_x = 0, b_y = 1, b_h = 2, b_w = 3;
        let box = Scale_Coordinates(w, h, bounding_box[cur_time]);
        c.rect(box[b_x], box[b_y],box[b_w], box[b_h]);
        c.lineWidth = 4;
        c.strokeStyle = 'red';
        c.stroke();
    }
    // Start over immediately since timeout is set to 0
    setTimeout(function(){ draw(v,c,bc,w,h); }, 0);
}

// FIXME: This is to test the functionality of the bounding box. The current box will be hardcoded for now.
//        In the coordinates will be extracted from database.
function Scale_Coordinates(cw, ch, coordinates){
    // cw -> canvas width, ch -> canvas height, coordinates -> [x, y, w, h] -> location of bounding box
    // if canvas width and height are equal to dimensions of video then the coordinates do not need to be scaled
    if(cw === 1280 && ch === 720){
        return coordinates;
    }
    // otherwise scale based on the canvas width and canvas height
    coordinates[0] = coordinates[0] * (cw / 1280); // Scaling x
    coordinates[1] = coordinates[1] * (ch / 720);  // Scaling y
    coordinates[2] = coordinates[2] * (cw / 1280); // Scaling w
    coordinates[3] = coordinates[3] * (ch / 720); // Scaling h
    return coordinates; // returns scaled coordinates for bounding box
}