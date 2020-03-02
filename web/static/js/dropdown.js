document.addEventListener('DOMContentLoaded', function(){


    document.getElementById("dog").addEventListener("click", function() {
            dropdown_menu('dog');
        }, false);


    document.getElementById("bike").addEventListener("click", function() {
            dropdown_menu('motorbike');
        }, false);


    document.getElementById("people").addEventListener("click", function() {
            dropdown_menu('person');
            }, false);

    document.getElementById("bicycle").addEventListener("click", function() {
            dropdown_menu('bicycle');
        }, false);


    document.getElementById("car").addEventListener("click", function() {
            dropdown_menu('car');
        }, false);


    document.getElementById("aeroplane").addEventListener("click", function() {
            dropdown_menu('aeroplane');
            }, false);

    document.getElementById("bus").addEventListener("click", function() {
            dropdown_menu('bus');
        }, false);


    document.getElementById("train").addEventListener("click", function() {
            dropdown_menu('train');
        }, false);


    document.getElementById("truck").addEventListener("click", function() {
            dropdown_menu('truck');
            }, false);

    document.getElementById("boat").addEventListener("click", function() {
            dropdown_menu('boat');
        }, false);


    document.getElementById("traffic light").addEventListener("click", function() {
            dropdown_menu('traffic light');
        }, false);


    document.getElementById("chair").addEventListener("click", function() {
            dropdown_menu('chair');
            }, false);


    function dropdown_menu(a){
        obj = a;
        change_dropdown(a);
    }


    function change_dropdown(a){
        document.getElementById("drop").textContent = a;
    }

},false);