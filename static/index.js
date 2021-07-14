document.querySelector(".finalBill").style.visibility = "hidden";

var totalAmt = 0;
function calculate(valu,qty) {
  totalAmt = totalAmt + valu*qty;
}

function showBill() {
  document.querySelector("#totalB").textContent = totalAmt;
  document.querySelector(".finalBill").style.visibility="visible";
  document.querySelector(".showB").style.visibility="hidden";
}


