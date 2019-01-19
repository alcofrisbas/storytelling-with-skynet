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

window.onload = load;
