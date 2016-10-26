// NOTE you have to configure this!
var _pEXTERNAL_ASSETS = "/external_assets";

document.writeln("<script src='" + _pEXTERNAL_ASSETS + "/javascript/appConfig.js'><\/script>");

// NOTE this is a global hack to get the zap viewers to work
document.writeln("<style>.simstripCont{white-space:nowrap}</style>");

/**
 * Protect window.console method calls, e.g. console is not defined on IE8, IE9
 * unless dev tools are open, and IE doesn't define console.debug
 * If 'console' is undefined, then it is defined and all possible console.* functions are defined,
 * but operationally do nothing. This prevents javascript exceptions in IE8, IE9.
 * Adapted from https://stackoverflow.com/questions/3326650/console-is-undefined-error-for-internet-explorer
 */
(function () {
    if (!window.console) {
        window.console = {};
    }
    // union of Chrome, FF, IE, and Safari console methods
    var m = [
        'assert', 'clear', 'count', 'debug', 'dir', 'dirxml', 'error',
        'exception', 'group', 'groupCollapsed', 'groupEnd', 'info', 'log',
        'markTimeline', 'profile', 'profileEnd', 'table', 'time', 'timeEnd',
        'timeStamp', 'trace', 'warn'
    ];
    // define undefined methods as noops to prevent errors
    for (var i = 0; i < m.length; i++) {
        if (!window.console[m[i]]) {
            window.console[m[i]] = function () {
            };
        }
    }
})();

//v1.7
// Flash Player Version Detection
// Detect Client Browser type
// Copyright 2005-2007 Adobe Systems Incorporated.  All rights reserved.
var isIE = (navigator.appVersion.indexOf("MSIE") != -1) ? true : false;
var isWin = (navigator.appVersion.toLowerCase().indexOf("win") != -1) ? true : false;
var isOpera = (navigator.userAgent.indexOf("Opera") != -1) ? true : false;

function ControlVersion() {
    var version;
    var axo;
    var e;

    // NOTE : new ActiveXObject(strFoo) throws an exception if strFoo isn't in the registry

    try {
        // version will be set for 7.X or greater players
        axo = new ActiveXObject("ShockwaveFlash.ShockwaveFlash.7");
        version = axo.GetVariable("$version");
    } catch (e) {
    }

    if (!version) {
        try {
            // version will be set for 6.X players only
            axo = new ActiveXObject("ShockwaveFlash.ShockwaveFlash.6");

            // installed player is some revision of 6.0
            // GetVariable("$version") crashes for versions 6.0.22 through 6.0.29,
            // so we have to be careful.

            // default to the first public version
            version = "WIN 6,0,21,0";

            // throws if AllowScripAccess does not exist (introduced in 6.0r47)
            axo.AllowScriptAccess = "always";

            // safe to call for 6.0r47 or greater
            version = axo.GetVariable("$version");

        } catch (e) {
        }
    }

    if (!version) {
        try {
            // version will be set for 4.X or 5.X player
            axo = new ActiveXObject("ShockwaveFlash.ShockwaveFlash.3");
            version = axo.GetVariable("$version");
        } catch (e) {
        }
    }

    if (!version) {
        try {
            // version will be set for 3.X player
            axo = new ActiveXObject("ShockwaveFlash.ShockwaveFlash.3");
            version = "WIN 3,0,18,0";
        } catch (e) {
        }
    }

    if (!version) {
        try {
            // version will be set for 2.X player
            axo = new ActiveXObject("ShockwaveFlash.ShockwaveFlash");
            version = "WIN 2,0,0,11";
        } catch (e) {
            version = -1;
        }
    }

    return version;
}

// JavaScript helper required to detect Flash Player PlugIn version information
function GetSwfVer() {
    // NS/Opera version >= 3 check for Flash plugin in plugin array
    var flashVer = -1;

    if (navigator.plugins != null && navigator.plugins.length > 0) {
        if (navigator.plugins["Shockwave Flash 2.0"] || navigator.plugins["Shockwave Flash"]) {
            var swVer2 = navigator.plugins["Shockwave Flash 2.0"] ? " 2.0" : "";
            var flashDescription = navigator.plugins["Shockwave Flash" + swVer2].description;
            var descArray = flashDescription.split(" ");
            var tempArrayMajor = descArray[2].split(".");
            var versionMajor = tempArrayMajor[0];
            var versionMinor = tempArrayMajor[1];
            var versionRevision = descArray[3];
            if (versionRevision == "") {
                versionRevision = descArray[4];
            }
            if (versionRevision[0] == "d") {
                versionRevision = versionRevision.substring(1);
            } else if (versionRevision[0] == "r") {
                versionRevision = versionRevision.substring(1);
                if (versionRevision.indexOf("d") > 0) {
                    versionRevision = versionRevision.substring(0, versionRevision.indexOf("d"));
                }
            }
            var flashVer = versionMajor + "." + versionMinor + "." + versionRevision;
        }
    }
    // MSN/WebTV 2.6 supports Flash 4
    else if (navigator.userAgent.toLowerCase().indexOf("webtv/2.6") != -1) flashVer = 4;
    // WebTV 2.5 supports Flash 3
    else if (navigator.userAgent.toLowerCase().indexOf("webtv/2.5") != -1) flashVer = 3;
    // older WebTV supports Flash 2
    else if (navigator.userAgent.toLowerCase().indexOf("webtv") != -1) flashVer = 2;
    else if (isIE && isWin && !isOpera) {
        flashVer = ControlVersion();
    }
    return flashVer;
}

// When called with reqMajorVer, reqMinorVer, reqRevision returns true if that version or greater is available
function DetectFlashVer(reqMajorVer, reqMinorVer, reqRevision) {
    var versionArray;
    var versionStr = GetSwfVer();
    if (versionStr == -1) {
        return false;
    } else if (versionStr != 0) {
        if (isIE && isWin && !isOpera) {
            // Given "WIN 2,0,0,11"
            var tempArray = versionStr.split(" ");  // ["WIN", "2,0,0,11"]
            var tempString = tempArray[1];           // "2,0,0,11"
            versionArray = tempString.split(",");  // ['2', '0', '0', '11']
        } else {
            versionArray = versionStr.split(".");
        }
        var versionMajor = versionArray[0];
        var versionMinor = versionArray[1];
        var versionRevision = versionArray[2];

        // is the major.revision >= requested major.revision AND the minor version >= requested minor
        if (versionMajor > parseFloat(reqMajorVer)) {
            return true;
        } else if (versionMajor == parseFloat(reqMajorVer)) {
            if (versionMinor > parseFloat(reqMinorVer))
                return true;
            else if (versionMinor == parseFloat(reqMinorVer)) {
                if (versionRevision >= parseFloat(reqRevision))
                    return true;
            }
        }
        return false;
    }
}
// END Flash Player Version Detection
/////////////////////////////////////

var _pBrowserSupport = {

    // initialize with an object containing keys & values for browser names & versions as in:
    // {webkit:'531.0', msie:'8.0', mozilla:'1.9.2'}
    //
    // this.supported means the current browser has version >= the minumum_list,
    // this.not_supported means the current browser is in the minumum_list and
    // has vesion less than that given.
    //
    // NOTE that both supported & not_supported may legitimately be false, if the
    // current browser is not in the list.
    initialize: function (minimum_list) {

        var userAgent = navigator.userAgent.toLowerCase();
        this.minimum_list = minimum_list ? minimum_list : {};

        this.version = (userAgent.match(/.+(?:rv|it|ra|ie|me)[\/: ]([\d.]+)/) || [])[1];

        this.webkit = /webkit/.test(userAgent);
        this.opera = /opera/.test(userAgent);
        this.msie = /msie|trident/.test(userAgent) && !/opera/.test(userAgent); // only IE uses trident rendering engine
        this.mozilla = /mozilla/.test(userAgent) && !/(trident|compatible|webkit)/.test(userAgent);
        this.chrome = /chrome/.test(userAgent);

        this.ie_compat_version = this.version; // is this used??

        // Note: IE browser version is based on the Trident rendering engine version number, which is used on post IE7 browsers
        // Trident info: https://en.wikipedia.org/wiki/Trident_(layout_engine)
        if (this.msie) {
            var calcVersion = this._browserVersionFromTrident(userAgent);
            this.version = calcVersion ? calcVersion : this.version;
        }

        this.extended_version = this._version_to_number(this.version);

        this.name = this.chrome ? 'chrome' : this.webkit ? 'webkit' : this.opera ? 'opera' : this.msie ? 'msie' : this.mozilla ? 'mozilla' : 'unknown';
        this.supported = this._is_supported(this.minimum_list);
        this.not_supported = this._is_not_supported(this.minimum_list);
    },

    /**
     * Determine IE browser version based on Trident rendering engine version.
     * This is effective beginning with IE8, Trident4
     *
     * For IE8, Trident4, the versions of both browser and trident have incremented by one each all the way
     * up to and including IE11, Trident7
     * We will use this algorithm until it fails someday in the future, at which point the business logic
     * can be adjusted.
     *
     * @param {string} userAgent downcased version of the user agent string
     * @returns {string} [version number].0, or undefined if trident number is too low
     */
    _browserVersionFromTrident: function (userAgent) {
        var ieStart = 8;
        var tridentStart = 4;
        var ieVersion;

        var tridentMatch = userAgent.match(/trident\/(\d+)/);
        if (tridentMatch) {
            var tridentNum = parseInt(tridentMatch[1]);
            if (tridentNum >= tridentStart) {
                var delta = tridentNum - tridentStart;
                ieVersion = (ieStart + delta) + ".0";
            }
        }

        return ieVersion;
    },

    _is_not_supported: function (list) {

        for (var key in list) {
            if (key == this.name) {
                var supported_version = this._version_to_number(list[key]);

                if (supported_version.major > this.extended_version.major)
                    return (true);

                if ((supported_version.major == this.extended_version.major)
                    &&
                    (supported_version.minor > this.extended_version.minor))
                    return (true);

                if ((supported_version.major == this.extended_version.major)
                    &&
                    (supported_version.minor == this.extended_version.minor)
                    &&
                    (supported_version.build > this.extended_version.build))
                    return (true);
            }
        }
        return (false);
    },

    _is_supported: function (list) {
        for (var key in list) {
            if (key == this.name) {
                var supported_version = this._version_to_number(list[key]);

                if ((supported_version.major == this.extended_version.major)
                    &&
                    (supported_version.minor == this.extended_version.minor)
                    &&
                    (supported_version.build <= this.extended_version.build))
                    return (true);

                if ((supported_version.major == this.extended_version.major)
                    &&
                    (supported_version.minor <= this.extended_version.minor))
                    return (true);

                if (supported_version.major <= this.extended_version.major)
                    return (true);
            }
        }
        return (false);
    },

    _version_to_number: function (version) {

        var version_components = version ? version.split('.') : [0];
        var ret = {};

        ret.major = parseInt(version_components[0]);
        ret.minor = version_components.length > 1 ? parseInt(version_components[1]) : 0;
        ret.build = version_components.length > 2 ? parseInt(version_components[2]) : 0;
        return (ret);
    },

    _cookie_check: function () {

        var tmpcookie = new Date();
        var chkcookie = (tmpcookie.getTime() + '');
        document.cookie = "chkcookie=" + chkcookie + "; path=/";
        if (document.cookie.indexOf(chkcookie, 0) < 0)
            return (false);

        return (true);
    },

    _flash_check: function () {

        if (!flashVersion || !flashVersion.length >= 3)
            return (false);

        if (!DetectFlashVer)
            return (false);

        return (DetectFlashVer(flashVersion[0], flashVersion[1], flashVersion[2]));
    },

    _flash_version: function () {

        if (!GetSwfVer)
            return ('unknown');

        return (GetSwfVer());
    }

};


function _pSiteWarning() {

    this.warn_box_zindex = 100000; // make sure above SIV, Dual Viewer SIV
    this.warn_box = null;
    this.warn_content = '';
    this.warning_present = false;

    this.show_stats = function () {

        var msg = "Browser name: " + _pBrowserSupport.name;
        msg += "<br/>Version: " + _pBrowserSupport.version;
        msg += "<br/>IE compat version: " + _pBrowserSupport.ie_compat_version;
        add_warning(msg);
    }

    // called by atlasviewer
    this.show_flash_warning = function () {
        if (!_pBrowserSupport.supported_flash) {

            var flash_ver = flashVersion ? 'Flash version ' + flashVersion.join('.') + ' or higher' : 'A more recent version of Flash';
            var warn_list = document.getElementById('_pWarnList');

            if (warn_list) {
                var li = document.createElement('li');
                li.style.cssText = 'margin-left:22px;';
                li.innerHTML = flash_ver + " is required.";
                warn_list.appendChild(li);
            } else {
                var msg = "<br/>This site requires " + flash_ver;
                msg += "<br/>Some content may not function properly.";
                add_warning(msg);
            }
        }
    }

    var _self = this;

    function create_warning_box() {

        _self.warn_box = document.createElement('div');
        _self.warn_box.setAttribute('id', 'version_warning_container');
        _self.warn_box.style.cssText = 'width:480px; height:74px; border:1px solid #f00; position:absolute; top:8px; left:260px;background-color:#fee;padding:6px;color:#c00' + ';z-index:' + _self.warn_box_zindex;
        document.body.appendChild(_self.warn_box);
    }

    function add_warning(msg) {

        if (!_self.warn_box)
            create_warning_box();

        // warn some other way?
        if (!_self.warn_box)
            return;

        _self.warn_content += msg;
        _self.warn_box.innerHTML = _self.warn_content;
    }

    function show_warning(browser_info) {

        var msg = "Your web browser does not meet one or more of the system requirements for this site:<ul id='_pWarnList' style='padding:0px; margin:0px;'>";

        if (!browser_info.supported)
            msg += "<li style='margin-left:22px;'>Your browser version is not supported.</li>";
        msg += "</ul>";

        if (!browser_info.supported) {

            // if the legacy warning is already present, hide it.
            var old_warn = document.getElementById('js_cookie_check');
            if (old_warn)
                old_warn.style.cssText = "display:none";

            msg += "<div style='margin-top:4px;'>To see the minimum requirements for this site click <a href='javascript:_pShowSysReqs();'>here</a>.</div>";
            _self.warning_present = true;
            add_warning(msg);
        }
    }

    function show_flash_warning() {

        if (typeof _pSupressBrowserFlashWarning !== 'undefined' && _pSupressBrowserFlashWarning)
            return;

        if (!_pBrowserSupport.supported_flash) {

            var flash_ver = flashVersion ? 'Flash version ' + flashVersion.join('.') + ' or higher' : 'A more recent version of Flash';
            var warn_list = document.getElementById('_pWarnList');

            if (warn_list) {
                var li = document.createElement('li');
                li.style.cssText = 'margin-left:22px;';
                li.innerHTML = flash_ver + " is required.";
                warn_list.appendChild(li);
            } else {
                var msg = "<br/>This site requires " + flash_ver;
                msg += "<br/>Some content may not function properly.";
                add_warning(msg);
            }
        }
    }

    function add_warning_closer() {

        if (!_self.warn_box)
            return;

        var style = "width:16px; height:16px; border:0px solid #a99; position:absolute; top:3px; right:3px; cursor:pointer";
        var src = _pEXTERNAL_ASSETS + "/images/close_x.png";
        _self.warn_content += "<img id='version_warning_closer' src='" + src + "' style='" + style + "'/>";
        _self.warn_box.innerHTML = _self.warn_content;

        var closer = document.getElementById('version_warning_closer');
        if (closer.addEventListener) {
            document.getElementById('version_warning_closer').addEventListener(
                'click',
                function () {
                    document.getElementById('version_warning_container').style.display = "none";
                },
                false);
        } else {
            document.getElementById('version_warning_closer').attachEvent(
                'onclick',
                function () {
                    document.getElementById('version_warning_container').style.display = "none";
                },
                false);
        }
    }

    function init() {

        _pBrowserSupport.initialize(_pSUPPORTED_BROWSERS);
        var url = document.URL;

//        if (url.indexOf('show_browser_stats') >= 0)
//            _self.show_stats();
//        else
//            show_warning(_pBrowserSupport);
//
//        show_flash_warning();
//
//        add_warning_closer();
    }

    init();
}

var _pSiteWarnings = null;
function _pShowFlashWarning() {


    if (!_pSiteWarnings)
        _pSiteWarnings = new _pSiteWarning();

    if (_pSiteWarnings)
        _pSiteWarnings.show_flash_warning();
}

function _pShowSysReqs() {

    var reqs_win = window.open('', '_blank', 'width=460,height=400,status=0,scrollbars=0,titlebar=0,location=0');
    reqs_win.document.writeln("<script src='" + _pEXTERNAL_ASSETS + "/javascript/browserVersions.js'><\/script>");
}

if (window.addEventListener)
    window.addEventListener('load', function () {
        if (!_pSiteWarnings) _pSiteWarnings = new _pSiteWarning();
    }, false);
else if (window.attachEvent)
    window.attachEvent('onload', function () {
        if (!_pSiteWarnings) _pSiteWarnings = new _pSiteWarning();
    });


/*
 * JavaScript used by the ABA Portal.
 *
 * Configuration of the Portal app is done by setting values of the four arrays,
 * _pTabNames, _pTabLinks, _pMoreProjectsMenu, _pFooterLinks
 *
 * Note on IE browser support - needed to put in a hack in _pGetPosOffset() for IE versions > v5
 * browers. May need to tweak hack for future versions of IE.
 */

//********************************************
//******** define constants for site search ********
//********************************************
var _pSiteSearchUrl = "http://host/search/index.html?query=";
var _pSiteSearchButton = "pSiteSearchButton";
var _pSiteSearchTextInput = "pSiteSearchTextInput";

//*****************************************************************
//****** define all tab names, this determines            ********
//****** what is displayed as tab text                    ********
//****** (index is tab CSS ID)                            ********
//****** Note: order of appearance of main menu items is  ********
//****** determined in portalHeader.js                    ********
//****************************************************************
var _pTabNames = [];
_pTabNames["pHome"] = "Home";
_pTabNames["pMouseBrain"] = "Mouse Brain";
_pTabNames["pDevelopingMouseBrain"] = "Developing Mouse Brain";
_pTabNames["pHuman"] = "Human Brain";
_pTabNames["pMouseConnectivity"] = "Mouse Connectivity";
_pTabNames["pMoreProjects"] = "More";
_pTabNames["pDevelopingHumanBrain"] = "Developing Human Brain";
_pTabNames["pSleep"] = "Sleep";
_pTabNames["pMouseDiversity"] = "Mouse Strains";
_pTabNames["pNonHumanPrimate"] = "Non-Human Primate";
_pTabNames["pMouseSpinalCord"] = "Mouse Spinal Cord";
_pTabNames["pGlioblastoma"] = "Glioblastoma";
_pTabNames["pCellTypes"] = "Cell Types";

_pTabNames["pAnnouncements"] = "Get Started";
_pTabNames["pTutorials"] = "Tutorials";
_pTabNames["pHighlights"] = "Data Highlights";
_pTabNames["pCaseStudies"] = "Case Study";
_pTabNames["pAPI"] = "API";
_pTabNames["pAtlas"] = "Reference Atlases";
_pTabNames["pHelp"] = "Help";


//*************************************
//****** define tab link urls  ********
//****** (index is tab CSS ID) ********
//*************************************
var _pTabLinks = [];
_pTabLinks["pHome"] = "http://www.brain-map.org/";
_pTabLinks["pMouseBrain"] = "http://mouse.brain-map.org/";
_pTabLinks["pMouseSpinalCord"] = "http://mousespinal.brain-map.org/";
_pTabLinks["pDevelopingMouseBrain"] = "http://developingmouse.brain-map.org/";
_pTabLinks["pHuman"] = "http://human.brain-map.org/";
_pTabLinks["pDevelopingHumanBrain"] = "http://www.brainspan.org/";
_pTabLinks["pMouseConnectivity"] = "http://connectivity.brain-map.org/";
_pTabLinks["pNonHumanPrimate"] = "http://www.blueprintnhpatlas.org/";
_pTabLinks["pGlioblastoma"] = "http://glioblastoma.alleninstitute.org/";
_pTabLinks["pCellTypes"] = "http://celltypes.alleninstitute.org/";

_pTabLinks["pAnnouncements"] = "http://www.brain-map.org/announcements/index.html";
_pTabLinks["pTutorials"] = "http://www.brain-map.org/tutorials/index.html";
_pTabLinks["pHighlights"] = "http://www.brain-map.org/highlights/index.html";
_pTabLinks["pCaseStudies"] = "http://casestudies.brain-map.org/ggb";
_pTabLinks["pAPI"] = "http://www.brain-map.org/api/index.html";
_pTabLinks["pAtlas"] = "http://atlas.brain-map.org";
_pTabLinks["pHelp"] = "http://help.brain-map.org";

// note: _pTabLinkTargets is a separate table because _pTabLinks values are accessed directly by some of the apps.
var _pTabLinkTargets = [];
_pTabLinkTargets["pHome"] = '_self';
_pTabLinkTargets["pMouseBrain"] = '_self';
_pTabLinkTargets["pMouseSpinalCord"] = '_self';
_pTabLinkTargets["pDevelopingMouseBrain"] = '_self';
_pTabLinkTargets["pHuman"] = '_self';
_pTabLinkTargets["pDevelopingHumanBrain"] = '_blank';
_pTabLinkTargets["pMouseConnectivity"] = '_self';
_pTabLinkTargets["pNonHumanPrimate"] = '_blank';
_pTabLinkTargets["pGlioblastoma"] = '_blank';
_pTabLinkTargets["pCellTypes"] = '_blank';

_pTabLinkTargets["pAnnouncements"] = '_self';
_pTabLinkTargets["pTutorials"] = '_self';
_pTabLinkTargets["pHighlights"] = '_self';
_pTabLinkTargets["pCaseStudies"] = '_blank';
_pTabLinkTargets["pAPI"] = '_self';
_pTabLinkTargets["pAtlas"] = '_blank';
_pTabLinkTargets["pHelp"] = '_self';

var _pIsTopLevelTab = {
    "pHome": true,
    "pAnnouncements": true,
    "pHelp": true
};

//**************************************************************************
//****** define the "More Projects" drop down menu items and links  ********
//****** 'items' property has array of values: name, link, target   ********
//****** The 'target' value is optional, can be '_blank', '_self'   ********
//
//       Note: this hash and array are probably no longer used
//**************************************************************************
var _pMoreProjectsMenu = {divclass: 'pDropDownMenu', inlinestyle: '', linktarget: '_self'};
_pMoreProjectsMenu.items = [
    [_pTabNames["pDevelopingHumanBrain"], _pTabLinks["pDevelopingHumanBrain"], "_blank"],
    [_pTabNames["pGlioblastoma"], _pTabLinks["pGlioblastoma"], "_blank"],
    [_pTabNames["pNonHumanPrimate"], _pTabLinks["pNonHumanPrimate"], "_blank"],
    [_pTabNames["pMouseSpinalCord"], _pTabLinks["pMouseSpinalCord"]],
    [_pTabNames["pMouseDiversity"], _pTabLinks["pMouseDiversity"]],
    [_pTabNames["pSleep"], _pTabLinks["pSleep"]]
];

//****************************************
//****** define the header links *********
//****************************************
var _pHeaderLinks = new Object();


//****************************************
//****** define the footer links *********
//****************************************
var _pFooterLinks = [];
_pFooterLinks["pPrivacyPolicy"] = "http://www.alleninstitute.org/Media/policies/privacy_policy_content.html";
_pFooterLinks["pTermsOfUse"] = "https://github.com/AllenInstitute/AllenSDK/blob/master/COPYING";
_pFooterLinks["pCitationPolicy"] = "http://www.alleninstitute.org/Media/policies/citation_policy_content.html";
_pFooterLinks["pAbout"] = "http://www.alleninstitute.org/about_us/overview.html";
_pFooterLinks["pContactUs"] = "http://www.alleninstitute.org/contact_us/index.html";
_pFooterLinks["pFooterLogo"] = "http://www.AllenInstitute.org/";

_pFooterLinks["pPublications"] = "http://alleninstitute.org/science/publications/index.html";
_pFooterLinks["pContactUs"] = "http://alleninstitute.org/contact_us/index.html";

_pFooterLinks["pFacebook"] = "http://facebook.com/AllenInstitute";
_pFooterLinks["pTwitter"] = "http://twitter.com/Allen_Institute";
_pFooterLinks["pYouTube"] = "http://www.youtube.com/AllenInstitute";
_pFooterLinks["pLinkedIn"] = "http://www.linkedin.com/company/allen-institute-for-brain-science";


/*
 * Checks to ensure global javascript vars _pImagePath, _pMoreProjectsId, _pTabId are defined,
 * then uses _pTabId to select the designated project tab.
 *
 * This function needs to be called directly from the host HTML page, in response to a JavaScript "onload"
 * event.
 *
 * Valid values of _pTabId are:
 *   pHome, pMouseBrain, pMouseSpinalCord, pDevelopingMouseBrain, pHumanBrain, pMoreProjects, pSleep, etc.
 */
function _pPortalOnLoad() {
    // ***** use javascript vars defined on main HTML page: _pTabId, _pMoreProjectsId, _pImagePath ********
    // ***** Note: pImagePath must end in a "/" ********

    var error;
    var theTab;

    // validate _pImagePath
    try {
        if (_pImagePath) {
            if (_pImagePath.charAt(_pImagePath.length - 1) != "/") {
                throw "noSlash";
            }
        }
        else if (_pImagePath == undefined) {
            throw "undefined";
        }
        else if (_pImagePath == "") {
            throw "emptyString";
        }
    }
    catch (error) {
        if (error == "noSlash") {
            alert("Javascript var _pImagePath needs to be terminated with a '/'");
        }
        else if (error == "emptyString") {
            alert("Javascript var _pImagePath is an empty String");
        }
        else {
            alert("Javascript var _pImagePath is undeclared or undefined");
        }
        return;
    }

    // validate _pMoreProjectsId
    try {
        if (_pMoreProjectsId) {
            // do nothing, it has a value
        }
        else if (_pMoreProjectsId == undefined) {
            throw "undefined";
        }
        else if (_pMoreProjectsId == "") {
            throw "emptyString";
        }
    }
    catch (error) {
        if (error == "emptyString") {
            alert("Javascript var _pMoreProjectsId is an empty String");
        }
        else {
            alert("Javascript var _pMoreProjectsId is undeclared or undefined");
        }
        return;
    }

    // validate _pMoreProjectsId
    try {
        if (_pTabId) {
            var tabName = _pTabNames[_pTabId];
            theTab = _pSetSelectedTab(_pTabId, "pTabSelected", tabName);
            if (theTab == null) {
                throw "null";
            }
        }
        else if (_pTabId == undefined) {
            throw "undefined";
        }
        else if (_pTabId == "") {
            throw "emptyString";
        }
    }
    catch (error) {
        if (error == "null") {
            alert("Element for menu item _pTabId = " + _pTabId + " was not found in the DOM");
        }
        else if (error == "emptyString") {
            alert("Javascript var _pTabId is an empty String");
        }
        else {
            alert("_pTabId is undeclared or undefined");
        }
        return;
    }
}


/*  
 Set the visual state of the current page's tab.
 NOTE that whenever we select a tab we have just loaded a new page,
 so I'm not worrying about UNselecting anything here.
 */
function _pSetSelectedTab(tab_id) {

    var tab = document.getElementById(tab_id);

    // nav item sare being treated differently depending on whether they are
    // top-level tabs or flyout menu items.
    if (_pIsTopLevelTab[tab_id]) {

        // top-level tabs are simple, add the class that
        // (currently) just changes the color of the text.
        tab.parentNode.className += " portal_tab_active";
    }
    else {

        // In this case we want to change the text of the menu header
        // to show the current selection, as well as highlight
        // the item that's selected in the flyout.
        tab.parentNode.className += " portal_menu_item_active";
        var title = document.getElementById("pMenuTitle");
        title.innerHTML = _pTabNames[tab_id];
        title.parentNode.className += " portal_menu_active";
    }

    return (tab);
}

/* turn "More Projects" triangle green on mouseover */
function _pTriangleMouseOver() {
    if (document.images)
        document._pTriangle.src = _pImagePath + "arrow_over.gif";
}

/* restore "More Projects" triangle image on mouseout */
function _pTriangleMouseOut() {
    if (document.images) {
        if (_pTabId == _pMoreProjectsId) {
            // white triangle (tab selected)
            document._pTriangle.src = _pImagePath + "arrow_on.gif";
        }
        else {
            // blue triangle (tab unselected)
            document._pTriangle.src = _pImagePath + "arrow_off.gif";
        }
    }
}

/* detect Enter/Return key press in Site Search Text input widget,
 * then simulate a click on the Site Search button.
 */
function doSiteSearchKeyPress(e) {
    // look for window.event in case event isn't passed in
    e = e || window.event;
    if (e.keyCode == 13) {
        document.getElementById(_pSiteSearchButton).click();
    }
}

/* build Site Search URL and open page */
function doSiteSearch() {
    var queryString;
    var searchInput = document.getElementById(_pSiteSearchTextInput);
    //auto select the passed qc facet
    var auto_select_facet = "&fa=false&e_sp=t&e_ag=t&e_tr=t&e_fa=t";

    if (searchInput != null) {
        queryString = searchInput.value;
    }

    if ((queryString != null) && (queryString.length > 0)) {
        location.href = _pSiteSearchUrl + encodeURIComponent(queryString) + auto_select_facet;
    }
    else {
        location.href = _pSiteSearchUrl + auto_select_facet;
    }

}
