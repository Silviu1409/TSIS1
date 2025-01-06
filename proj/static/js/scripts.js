document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('uploadForm');
    const fileInput = document.getElementById('file');

    uploadForm.addEventListener('submit', (event) => {
        if (fileInput.files.length === 0) {
            event.preventDefault();
            alert('Please select at least one file to upload!');
        }
    });
});

// window.onload = function() {
//     // Only run the script if we are on the "stats" page
//     if (window.location.pathname === "/stats") {
//         // Loop through each network visualization (iframe)
//         const networks = JSON.parse(localStorage.getItem("networks") || "[]");

//         networks.forEach(network => {
//             const networkId = network.name.replace(' ', '_');
//             const timestamp = network.timestamp;
//             const iframe = document.getElementById("iframe-" + networkId);
            
//             // Check if the network timestamp is already stored in localStorage
//             const cachedTimestamp = localStorage.getItem(networkId);
            
//             // If cached timestamp is not available or the timestamp has changed, reload the iframe
//             if (!cachedTimestamp || cachedTimestamp != timestamp) {
//                 iframe.src = iframe.src;  // Forces the iframe to reload the content
//                 localStorage.setItem(networkId, timestamp);  // Update the cached timestamp
//             }
//         });
//     }
// };

// // Save networks info in localStorage for reference on page load
// function setNetworksInfo(networks) {
//     localStorage.setItem("networks", JSON.stringify(networks));
// }