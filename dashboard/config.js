const localHosts = ['localhost', '127.0.0.1'];

window.FRAUD_API_URL = localHosts.includes(window.location.hostname)
    ? `${window.location.protocol}//${window.location.hostname}:8000`
    : 'https://fraud-detection-0c4l.onrender.com';