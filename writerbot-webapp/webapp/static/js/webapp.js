// JS for highlighting menu items
// a bit hacky, but effective.
// will patch with jquery as needed.
function load() {
    var url = window.location.pathname;
    console.log("load event detected: "+url);
    var hlt = document.getElementById(url);
    console.log(hlt);
    hlt.classList.add("active");
}

function showSaveInfo() {
    console.log("save pressed")
    var lst = document.getElementsByClassName("final");

    var i;
    for (i = 0; i < lst.length; i++) {
        if (lst[i].style.display === "none") {
          lst[i].style.display = "inline";
        } else {
          lst[i].style.display = "none";
        }
    }
}





function editUpdate(){
    var edit = document.getElementById("para").contentEditable
    console.log(edit)
    console.log("asdfasdf")
    if (edit == "false"){
        console.log("not anymore")
        document.getElementById("para").contentEditable = true;
        document.getElementById("para").style.border = "1px dotted";
        document.getElementById("para").style.padding = "0px";
        document.getElementById("edit-button").innerHTML = 'Update  <span class="oi oi-loop-circular"></span>';

    } else {
        console.log("edity")
        document.getElementById("para").contentEditable = false;
        document.getElementById("para").style.border = " none";
        document.getElementById("para").style.padding = "1px";
        document.getElementById("edit-button").innerHTML = 'Edit <span class="oi oi-pencil"></span>';
    }
    document.getElementById("edit-button").classList.toggle('btn-success');
    document.getElementById("edit-button").classList.toggle('btn-outline-primary');

}

function init(){
    console.log("listening?");
    document.getElementById('form').onsubmit = editUpdate;
}

window.onload = init;
window.onload = load;


$(document).ready(function(){
  $('[data-toggle="tooltip"]').tooltip();
});
