document.addEventListener('DOMContentLoaded', function(){

    document.getElementById("bike").addEventListener("click", function() {
            dropdown_menu('motorbike');
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