document.querySelector(".show").style.visibility = "hidden";
document.querySelector(".btn").onclick = function () {
    totalBill();
};
function totalBill() {
    var amount = document.querySelector(".amt").value;
    var rating = document.querySelector(".rate").value;
    var person = document.querySelector(".divideBill").value;

    var tip = amount * rating;
    var total = Number(amount) + Number(tip);
    total = total.toFixed(2);
    var tipPerPerson = tip / person;
    tipPerPerson = tipPerPerson.toFixed(2);

    document.querySelector("#totalValue").textContent = total;
    document.querySelector("#tipValue").textContent = tipPerPerson;
    document.querySelector(".show").style.visibility = "visible";
}
