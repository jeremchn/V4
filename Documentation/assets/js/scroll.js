function debounce(func, delay) {
    let timeoutId;
    return function () {
        const context = this;
        const args = arguments;
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => {
            func.apply(context, args);
        }, delay);
    };
}
// Function to check if an element is in the viewport
function isInViewport(elem) {
    const bounding = elem.getBoundingClientRect();
    return (
        bounding.top <= (window.innerHeight || document.documentElement.clientHeight) &&
        bounding.bottom >= 0 &&
        bounding.left <= (window.innerWidth || document.documentElement.clientWidth) &&
        bounding.right >= 0
    );
}
// Find the element in the viewport by ID
function findElementInViewportById(id) {
    const elem = document.getElementById(id);
    if (elem && isInViewport(elem)) {
        return elem;
    }
    return null; // Element not found or not in viewport
}
function handleScroll(event) {
    let addedTrue = false;
    const sections = document.querySelectorAll(".nav-link1");
    for (let i = 0; i < sections.length; i++) {
        const elem = sections[i];
        const val = elem.getAttribute("href");
        if (val != "javascript:void(0);" && val !== "#") {
            const elementInViewport = findElementInViewportById(val.slice(1));
            if (elementInViewport && !addedTrue) {
                elem.classList.add("active");
                addedTrue = true;
            } else {
                elem.classList.remove("active");
            }
        }
    }
}
// Debounce the scroll event listener
const debouncedHandleScroll = debounce(handleScroll, 100); // Adjust debounce delay as needed

// Add event listener with debounced function
window.addEventListener("scroll", debouncedHandleScroll);
debouncedHandleScroll();