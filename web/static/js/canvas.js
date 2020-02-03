// FIXME: need to extract coordinates from database
//  need to use these coordinates to draw bounding boxes against object
//  need to scale these coordinates based on the image size that is projected on the web page since its auto fitted


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
    if(v.paused || v.ended) return false;
    // First, draw it into the backing canvas
    bc.drawImage(v,0,0,w,h);
    // Grab the pixel data from the backing canvas
    var idata = bc.getImageData(0,0,w,h);
    var data = idata.data;
    // Loop through the pixels, turning them grayscale
    // for(var i = 0; i < data.length; i+=4) {
    //     var r = data[i];
    //     var g = data[i+1];
    //     var b = data[i+2];
    //     var brightness = (3*r+4*g+b)>>>3;
    //     data[i] = brightness;
    //     data[i+1] = brightness;
    //     data[i+2] = brightness;
    // }

    idata.data = data;
    // Draw the pixels onto the visible canvas
    c.putImageData(idata,0,0);
    c.beginPath();
    c.rect(188, 50, 200, 100);
    // c.fillStyle = 'yellow';
    // c.fill();
    c.lineWidth = 4;
    c.strokeStyle = 'red';
    c.stroke();
    // Start over!
    setTimeout(function(){ draw(v,c,bc,w,h); }, 0);
}
