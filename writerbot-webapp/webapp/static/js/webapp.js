var generation_mode = 1;
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


function setEdit(){
    console.log("editing!");
    var editor = document.getElementById("para");
    editor.contentEditable = true;
    editor.focus();
}


$("#para").keyup(function(event) {
    //console.log(event.keyCode);
});


$("#para").focusout(function(event) {
    var editor = document.getElementById("para");
    console.log(editor.textContent);
    $("#side-form").submit(
        // this makes it so we can submit extra values for django purposes
    function(event) {
        console.log("adding");
      $('<input />').attr('type', 'hidden')
          .attr('name', "sentence-content")
          .attr('value', editor.textContent)
          .appendTo('#side-form');
      return true;
  });
  $("#side-form").submit();
});


function editUpdate(){
    var edit = document.getElementById("para").contentEditable;
    var editButton = document.getElementById("edit-button");
    if (edit == "false"){
        document.getElementById("para").contentEditable = true;
        document.getElementById("para").style.border = "1px dotted";
        document.getElementById("para").style.padding = "0px";
        document.getElementById("edit-button").type = "button";
        document.getElementById("edit-button").innerHTML= '<i class="fas fa-pen"></i>';
    } else {
        document.getElementById("para").contentEditable = false;
        document.getElementById("para").style.border = " none";
        document.getElementById("para").style.padding = "1px";
        document.getElementById("edit-button").type = "submit";
        document.getElementById("edit-button").innerHTML= '<i class="fas fa-pencil-alt"></i>';
        // document.getElementById("side-form").submit();
    }
    //document.getElementById("edit-button").classList.toggle('btn-success');
    //document.getElementById("edit-button").classList.toggle('btn-outline-primary');
    console.log(editButton.type)
}


function togglePower(){
    console.log("power")
    var powerButton = document.getElementById("side-toggle-icon");
    powerButton.classList.toggle("glow");
    document.getElementById('side-toggle').name="side-toggle";
    document.getElementById('side-toggle').value="true";
    $("#side-form").submit(
        // this makes it so we can submit extra values for django purposes
);
}


//window.onload = init;
window.onload = load;


$(document).ready(function(){
  $('[data-toggle="tooltip"]').tooltip();
  document.getElementById('input-form').scrollIntoView(false);
});



$("#input-form").keydown(function(event)
{
    if (event.keyCode == 13)
    {
        event.preventDefault();
        $("#input-form").submit();
        document.getElementById('input-form').focus();
    }
});


$("#title-form").keyup(function(event)
{
    if (event.keyCode == 13)
    {
        event.preventDefault();
        $("#title-form").submit();
    }
});

function update_settings(mode){
    console.log("setting Mode: "+mode);
    generation_mode = mode;
    console.log(generation_mode);
}

function change_setting_text(){
    console.log(generation_mode);

    if (generation_mode == 1){
        document.getElementById("rnn-settings-title").style.display = "block";
        document.getElementById("ngram-settings-title").style.display = "none";
        document.getElementById("none-settings-title").style.display = "none";

        document.getElementById("rnn-settings-content").style.display = "block";
        document.getElementById("ngram-settings-content").style.display = "none";
        document.getElementById("none-settings-content").style.display = "none";
    }
    if (generation_mode == 2){
        document.getElementById("rnn-settings-title").style.display = "none";
        document.getElementById("ngram-settings-title").style.display = "block";
        document.getElementById("none-settings-title").style.display = "none";

        document.getElementById("rnn-settings-content").style.display = "none";
        document.getElementById("ngram-settings-content").style.display = "block";
        document.getElementById("none-settings-content").style.display = "none";
    }
    if (generation_mode == 3){
        document.getElementById("rnn-settings-title").style.display = "none";
        document.getElementById("ngram-settings-title").style.display = "none";
        document.getElementById("none-settings-title").style.display = "block";

        document.getElementById("rnn-settings-content").style.display = "none";
        document.getElementById("ngram-settings-content").style.display = "none";
        document.getElementById("none-settings-content").style.display = "block";
    }
}

$('input[name=mode]').change(function(){
    console.log("submitting settings");
    if (document.getElementById('rnn_mode').checked) {
        //console.log("rnn_mode checked");
        generation_mode = 1;
    }
    if (document.getElementById('ngram_mode').checked) {
        //console.log("rnn_mode checked");
        generation_mode = 2;
    }
    if (document.getElementById('none_mode').checked) {
        //console.log("rnn_mode checked");
        generation_mode = 3;
    }
    console.log(generation_mode);

    change_setting_text();
});
