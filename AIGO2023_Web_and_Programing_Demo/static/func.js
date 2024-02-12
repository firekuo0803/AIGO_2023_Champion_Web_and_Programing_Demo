var page2 = document.getElementById("page2");
var text = document.getElementById("text");
var keys = [];
var imgs = [];
var num = 0;
var dir = {}
//                  src="../static/public/rectangle-40@2x.png"

function test(n){
    var name = keys[n];
    var num3 = dir[keys[n]];
    var hj = num3[0];
    var path = num3[1];
    var img_num = path.length;

    text.innerHTML = `
        <div class="apple10">${name}</div>
        <div class="apple10">num: ${hj}</div>
    `;
    for (var s = 0; s<img_num; s++){
        page2.innerHTML = page2.innerHTML + `
            <div class="rectangle-wrapper" onclick = "location.href='../static/imgs/${name}/${path[s]}'">
                <img
                  class="instance-child7"
                  alt=""

                  src = "../static/imgs/${name}/${path[s]}"
                />
            </div>
        `;
    }

    var popup = document.getElementById("frameLink");
    if (!popup) return;
    var popupStyle = popup.style;
    if (popupStyle) {
    popupStyle.display = "flex";
    popupStyle.zIndex = 100;
    popupStyle.backgroundColor = "rgba(113, 113, 113, 0.3)";
    popupStyle.alignItems = "center";
    popupStyle.justifyContent = "center";
    }
    popup.setAttribute("closable", "");

    var onClick =
    popup.onClick ||function (e) {
      if (e.target === popup && popup.hasAttribute("closable")) {
        popupStyle.display = "none";
        page2.innerHTML = "";
        text.innerHTML = "";
      }
    };
    popup.addEventListener("click", onClick);

}

// 初始載入網頁時執行一次，然後每秒執行一次


function updateDataDisplay() {
// 使用 AJAX 請求讀取 data.json 檔案
    fetch('/static/data.json')
    .then(response => response.json())
    .then(data => {
        var list = document.getElementById("list");
        list.innerHTML = ""

        dir = data;
          keys = [];
          for(var key in dir){
              keys.push(key);
          }
          num = keys.length;

          for (var i = 0; i<num; i++){
              list.innerHTML = list.innerHTML + `
                  <div class="frame-parent" id="frameContainer${i}" onclick = "test(${i})">
                    <div class="apple-parent">
                      <div class="apple">${keys[i]}</div>
                      <div class="apple">num: ${dir[keys[i]][0]}</div>
                    </div>
                    <img
                      class="instance-child"
                      alt=""
                      src="../static/imgs/${keys[i]}/${dir[keys[i]][1][0]}"
                    />
                  </div>
              `;
          }

    });
}
updateDataDisplay();
setInterval(updateDataDisplay, 1000);
