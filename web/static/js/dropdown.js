document.addEventListener('DOMContentLoaded', function(){


    document.getElementById("dog").addEventListener("click", function() {
            dropdown_menu('dog');
        }, false);


    document.getElementById("bike").addEventListener("click", function() {
            dropdown_menu('bicycle');
        }, false);


    document.getElementById("people").addEventListener("click", function() {
            dropdown_menu('person');
            }, false);


    function dropdown_menu(a){
        obj = a;
        change_dropdown(a);
    }


    function change_dropdown(a){
        document.getElementById("drop").textContent = a;
    }

},false);