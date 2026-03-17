// 1. Open Webull in Chrome: https://app.webull.com/
// 2. Login to your account.
// 3. Right-click anywhere -> Inspect -> Console.
// 4. Paste the following and press Enter:

(function () {
  const tokens = {
    accessToken: localStorage.getItem('accessToken'),
    refreshToken: localStorage.getItem('refreshToken'),
    uuid: localStorage.getItem('uuid'),
    did: localStorage.getItem('did')
  };
  console.log("--- WEBULL TOKENS ---");
  console.log(JSON.stringify(tokens, null, 2));
  console.log("--------------------");
})();
