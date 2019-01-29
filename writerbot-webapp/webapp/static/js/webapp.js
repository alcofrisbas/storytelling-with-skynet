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

function init(){
    console.log("listening?");
    document.getElementById('form').onsubmit = editUpdate;
}

function togglePower(){
    console.log("power")
    var powerButton = document.getElementById("side-toggle-icon");
    powerButton.classList.toggle("glow");
    document.getElementById('side-toggle').name="side-toggle";
    document.getElementById('side-toggle').value="true";
    $("#side-form").submit(
        // this makes it so we can submit extra values for django purposes
  //   function(eventObj) {
  //     $('<input />').attr('type', 'hidden')
  //         .attr('name', "something")
  //         .attr('value', "something")
  //         .appendTo('#side-form');
  //     return true;
  // }
);
}

// $("#side-form").submit( function(eventObj) {
//       $('<input />').attr('type', 'hidden')
//           .attr('name', "something")
//           .attr('value', "something")
//           .appendTo('#side-form');
//       return true;
//   });


window.onload = init;
window.onload = load;


$(document).ready(function(){
  $('[data-toggle="tooltip"]').tooltip();
});

$("#para").keyup(function(event)
{
    console.log("teste alsd;fjk");
    if (event.keyCode == 13)
    {
        console.log(document.getElementById("para").innerHTML);
        event.preventDefault();
    //     $("#side-form").submit(
    //          function(eventObj) {
    //         $('<input />').attr('type', 'hidden')
    //                 .attr('name', "something")
    //                 .attr('value', "something")
    //                 .appendTo('#side-form');
    //         return true;
    //     }
    // );
    }
    return true;
});

$("#input-form").keyup(function(event)
{
    console.log("asdf");
    if (event.keyCode == 13)
    {
        event.preventDefault();
        $("#input-form").submit();
    }
    return true;
});

// document.addEventListener('keyup', function(e) {
//   if (e.key === 'Enter' && e.target.tagName === 'P') {
//     console.log('ENTER');
//     $("#side-form").submit(
//     //      function(eventObj) {
//     //     $('<input />').attr('type', 'hidden')
//     //             .attr('name', "something")
//     //             .attr('value', "something")
//     //             .appendTo('this');
//     //     return true;
//     // }
//     );
//   }
// });
