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
window.onload = load;
