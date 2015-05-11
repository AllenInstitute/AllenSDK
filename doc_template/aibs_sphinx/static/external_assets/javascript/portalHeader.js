
/* loads header HTML */
function import_portal()  {

    var header_content = "<style>" + 
"" + 
"	/* the icon font-family definition and helper classes */" + 
"	@font-face {" + 
"	  font-family: 'portal_icon_font';" + 
"	  src: url('http://ibs-timd-ux2/sites/assets/stylesheets/portal_icon_font.eot?39505677');" + 
"	  src: url('http://ibs-timd-ux2/sites/assets/stylesheets/portal_icon_font.eot?39505677#iefix') format('embedded-opentype')," + 
"	       url('http://ibs-timd-ux2/sites/assets/stylesheets/portal_icon_font.woff?39505677') format('woff')," + 
"	       url('http://ibs-timd-ux2/sites/assets/stylesheets/portal_icon_font.ttf?39505677') format('truetype')," + 
"	       url('http://ibs-timd-ux2/sites/assets/stylesheets/portal_icon_font.svg?39505677#portal_icon_font') format('svg');" + 
"	  font-weight: normal;" + 
"	  font-style: normal;" + 
"	}" + 
"	 " + 
"	 [class^=\"icon-\"]:before, [class*=\" icon-\"]:before {" + 
"	  font-family: \"portal_icon_font\";" + 
"	  font-style: normal;" + 
"	  font-weight: normal;" + 
"	  speak: none;" + 
"	 " + 
"	  display: inline-block;" + 
"	  text-decoration: inherit;" + 
"	  width: 1em;" + 
"	  margin-right: .2em;" + 
"	  text-align: center;" + 
"	  /* opacity: .8; */" + 
"	 " + 
"	  /* For safety - reset parent styles, that can break glyph codes*/" + 
"	  font-variant: normal;" + 
"	  text-transform: none;" + 
"	     " + 
"	  /* fix buttons height, for twitter bootstrap */" + 
"	  line-height: 1em;" + 
"	 " + 
"	  /* Animation center compensation - margins should be symmetric */" + 
"	  /* remove if not needed */" + 
"	  margin-left: .2em;" + 
"	" + 
"	}" + 
"	 " + 
"	.icon-spin5:before { content: '\\e806'; } /* '' */" + 
"	.icon-left-open-1:before { content: '\\e81e'; } /* '' */" + 
"	.icon-plus-1:before { content: '\\e801'; } /* '' */" + 
"	.icon-facebook:before { content: '\\e816'; } /* '' */" + 
"	.icon-gplus:before { content: '\\e818'; } /* '' */" + 
"	.icon-linkedin-3:before { content: '\\e815'; } /* '' */" + 
"	.icon-twitter:before { content: '\\e817'; } /* '' */" + 
"	.icon-youtube-play:before { content: '\\e805'; } /* '' */" + 
"	.icon-plus-2:before { content: '\\e809'; } /* '' */" + 
"	.icon-record-1:before { content: '\\e81c'; } /* '' */" + 
"	.icon-cancel:before { content: '\\e80b'; } /* '' */" + 
"	.icon-search:before { content: '\\e804'; } /* '' */" + 
"	.icon-cancel-2:before { content: '\\e80c'; } /* '' */" + 
"	.icon-plus-4:before { content: '\\e80d'; } /* '' */" + 
"	.icon-cancel-3:before { content: '\\e80e'; } /* '' */" + 
"	.icon-plus:before { content: '\\e808'; } /* '' */" + 
"	.icon-down-open:before { content: '\\e800'; } /* '' */" + 
"	.icon-right-open:before { content: '\\e802'; } /* '' */" + 
"	.icon-up-open:before { content: '\\e803'; } /* '' */" + 
"	.icon-stop:before { content: '\\e813'; } /* '' */" + 
"	.icon-cancel-1:before { content: '\\e820'; } /* '' */" + 
"	.icon-plus-3:before { content: '\\e80a'; } /* '' */" + 
"	/* end icon font helpers */" + 
"" + 
"	/* icon font spin animation */" + 
"	.animate-spin {" + 
"	  -moz-animation: spin 2s infinite linear;" + 
"	  -o-animation: spin 2s infinite linear;" + 
"	  -webkit-animation: spin 2s infinite linear;" + 
"	  animation: spin 2s infinite linear;" + 
"	  display: inline-block;" + 
"	}" + 
"	@-moz-keyframes spin {" + 
"	  0% {" + 
"	    -moz-transform: rotate(0deg);" + 
"	    -o-transform: rotate(0deg);" + 
"	    -webkit-transform: rotate(0deg);" + 
"	    transform: rotate(0deg);" + 
"	  }" + 
"" + 
"	  100% {" + 
"	    -moz-transform: rotate(359deg);" + 
"	    -o-transform: rotate(359deg);" + 
"	    -webkit-transform: rotate(359deg);" + 
"	    transform: rotate(359deg);" + 
"	  }" + 
"	}" + 
"	@-webkit-keyframes spin {" + 
"	  0% {" + 
"	    -moz-transform: rotate(0deg);" + 
"	    -o-transform: rotate(0deg);" + 
"	    -webkit-transform: rotate(0deg);" + 
"	    transform: rotate(0deg);" + 
"	  }" + 
"" + 
"	  100% {" + 
"	    -moz-transform: rotate(359deg);" + 
"	    -o-transform: rotate(359deg);" + 
"	    -webkit-transform: rotate(359deg);" + 
"	    transform: rotate(359deg);" + 
"	  }" + 
"	}" + 
"	@-o-keyframes spin {" + 
"	  0% {" + 
"	    -moz-transform: rotate(0deg);" + 
"	    -o-transform: rotate(0deg);" + 
"	    -webkit-transform: rotate(0deg);" + 
"	    transform: rotate(0deg);" + 
"	  }" + 
"" + 
"	  100% {" + 
"	    -moz-transform: rotate(359deg);" + 
"	    -o-transform: rotate(359deg);" + 
"	    -webkit-transform: rotate(359deg);" + 
"	    transform: rotate(359deg);" + 
"	  }" + 
"	}" + 
"	@-ms-keyframes spin {" + 
"	  0% {" + 
"	    -moz-transform: rotate(0deg);" + 
"	    -o-transform: rotate(0deg);" + 
"	    -webkit-transform: rotate(0deg);" + 
"	    transform: rotate(0deg);" + 
"	  }" + 
"" + 
"	  100% {" + 
"	    -moz-transform: rotate(359deg);" + 
"	    -o-transform: rotate(359deg);" + 
"	    -webkit-transform: rotate(359deg);" + 
"	    transform: rotate(359deg);" + 
"	  }" + 
"	}" + 
"	@keyframes spin {" + 
"	  0% {" + 
"	    -moz-transform: rotate(0deg);" + 
"	    -o-transform: rotate(0deg);" + 
"	    -webkit-transform: rotate(0deg);" + 
"	    transform: rotate(0deg);" + 
"	  }" + 
"" + 
"	  100% {" + 
"	    -moz-transform: rotate(359deg);" + 
"	    -o-transform: rotate(359deg);" + 
"	    -webkit-transform: rotate(359deg);" + 
"	    transform: rotate(359deg);" + 
"	  }" + 
"	}" + 
"	/*  end spin animation */	" + 
"" + 
"	@font-face {" + 
"	    font-family: 'bebas_neueregular';" + 
"	    src: url('http://ibs-timd-ux2/sites/assets/stylesheets/bebasneue-webfont.eot');" + 
"	    src: url('http://ibs-timd-ux2/sites/assets/stylesheets/bebasneue-webfont.eot?#iefix') format('embedded-opentype')," + 
"	         url('http://ibs-timd-ux2/sites/assets/stylesheets/bebasneue-webfont.woff') format('woff')," + 
"	         url('http://ibs-timd-ux2/sites/assets/stylesheets/bebasneue-webfont.ttf') format('truetype')," + 
"	         url('http://ibs-timd-ux2/sites/assets/stylesheets/bebasneue-webfont.svg#bebas_neueregular') format('svg');" + 
"	    font-weight: normal;" + 
"	    font-style: normal;" + 
"	}" + 
"" + 
"	/* top chrome styles */" + 
"	#top_chrome {" + 
"" + 
"		width:100%;" + 
"		height:28px;" + 
"		background-color:#163E65;" + 
"	}" + 
"" + 
"	.chrome_link_block {" + 
"" + 
"		color:#fff !important;	" + 
"		display:inline-block;" + 
"		height:100%;" + 
"		max-height:28px;" + 
"	}" + 
"" + 
"	.chrome_link_block:hover {" + 
"" + 
"		background-color:#eee;" + 
"		color:#0779BE !important;	" + 
"		cursor:pointer;" + 
"	}" + 
"" + 
"	#top_chrome a {" + 
"" + 
"		color:inherit !important;	" + 
"		text-decoration:none;" + 
"	}" + 
"" + 
"	#top_chrome a:hover {" + 
"" + 
"		color:inherit !important;	" + 
"	}" + 
"" + 
"	#top_chrome_buttons {" + 
"" + 
"		float:right;" + 
"		margin-right:50px;" + 
"	}" + 
"" + 
"	.chrome_button {" + 
"" + 
"		font-family:arial,sans-serif;" + 
"		font-weight:bold;" + 
"		display:inline-block;" + 
"		font-size:9px;" + 
"		text-align:left;" + 
"		text-decoration:none;" + 
"		margin:10px 20px 10px 20px;" + 
"	}" + 
"" + 
"	.chrome_active {" + 
"" + 
"		background-color:#1b7abd;" + 
"		color:#fff;" + 
"	}" + 
"" + 
"" + 
"	/* header element styles*/" + 
"	#header_content {" + 
"" + 
"		font-family: bebas_neueregular, arial, sans-serif;" + 
"		font-size:22px;" + 
"		color:#0779be;		" + 
"		letter-spacing:0.03em		;" + 
"" + 
"		position:relative;" + 
"		width:calc(100% - 112px);" + 
"		height:76px;" + 
"		border-bottom:2px solid #A5A8AB;" + 
"		margin-left:50px;" + 
"		margin-right:50px;" + 
"	}" + 
"" + 
"	#portal_logo {" + 
"" + 
"		position:absolute;" + 
"		top:20px;" + 
"		left:0px;" + 
"		width:236px;" + 
"	}" + 
"" + 
"	#portal_tabs {" + 
"" + 
"		position: absolute;" + 
"		bottom: 12px;" + 
"		left: 0px;" + 
"		margin-bottom: -12px;" + 
"		display: block;" + 
"	}" + 
"" + 
"	.horizontal_list" + 
"	{" + 
"		margin: 0;" + 
"		padding: 0;" + 
"		list-style-type: none;" + 
"		display: inline; 		" + 
"	}" + 
"" + 
"	.horizontal_list li { " + 
"" + 
"		display: inline; " + 
"		cursor:pointer;		" + 
"	}	" + 
"" + 
"	#portal_tab_menu_container {" + 
"" + 
"		display:none;" + 
"	}" + 
"" + 
"	#portal_search_input_group {" + 
"" + 
"		position:absolute;" + 
"		margin: 0;" + 
"		padding: 0;" + 
"		bottom:12px;" + 
"		right:0px;" + 
"		border:1px solid #DAD9D3;" + 
"		color:#DAD9D3;" + 
"	}" + 
"" + 
"	#portal_search_input_group input {" + 
"" + 
"		margin: 0;" + 
"		padding: 0;" + 
"		border:none;" + 
"		width:110px;" + 
"		color:#DAD9D3;" + 
"		font-family:Helvetica,sans-serif;" + 
"		font-size:12px;" + 
"		height:28px;" + 
"		margin-left:10px;" + 
"" + 
"		outline:none;" + 
"	}" + 
"" + 
"	#portal_search_input_group i {" + 
"" + 
"		font-size:0.8em;" + 
"		cursor:pointer;" + 
"	}" + 
"" + 
"	#pSiteSearchTextInput:focus {" + 
"" + 
"		color:#000;" + 
"	}" + 
"" + 
"	.icon_menu_down {" + 
"" + 
"		font-size: 12px;" + 
"		margin-left: 4px;" + 
"		vertical-align: text-top;" + 
"		padding-top: 4px;" + 
"		display: inline-block;" + 
"	}" + 
"" + 
"	.portal_tab_item {" + 
"" + 
"		background-color: #fff;" + 
"		padding: 12px;" + 
"		padding-left:20px;" + 
"		padding-right:20px;" + 
"		display: inline-block !important;		" + 
"	}" + 
"" + 
"	.portal_tab_item a:hover {" + 
"" + 
"		color:#fbe300;	" + 
"	}" + 
"" + 
"	.portal_tab_item:hover {" + 
"" + 
"		background-color: #0779be;	" + 
"		color:#fbe300 !important;	" + 
"	}	" + 
"" + 
"" + 
"	.portal_menu_active {" + 
"" + 
"		background-color: #EEEEEE;" + 
"	}	" + 
"" + 
"	.portal_menu_item_active {" + 
"" + 
"		color:#fbe300 !important;	" + 
"	}" + 
"" + 
"	.portal_tab_active {" + 
"" + 
"		color:#a5a8ab !important;" + 
"	}" + 
"" + 
"	#header_content a {		" + 
"" + 
"		color:inherit !important;" + 
"	}" + 
"" + 
"	#header_content a:hover {" + 
"" + 
"		background-color: #0779be;	" + 
"		color:#fbe300 !important;	" + 
"	}" + 
"" + 
"	#header_content ul {" + 
"" + 
"		list-style: none;" + 
"		position: relative;" + 
"		top: 58px;" + 
"		display: inline-table;" + 
"	}" + 
"" + 
"	#header_content ul ul {" + 
"		" + 
"		display:none;" + 
"		opacity:0;" + 
"	}" + 
"" + 
"	/* the fly out menus */" + 
"	#header_content ul li:hover > ul {" + 
"" + 
"		position: absolute;" + 
"		top: 48px;" + 
"		z-index:9999;" + 
"" + 
"		display: block;" + 
"		width: 152px;" + 
"		padding:0px;" + 
"		padding-top:20px;" + 
"		padding-bottom:10px;" + 
"		font-size: 12px;" + 
"		font-family: Arial,Sans-Serif;" + 
"		font-weight: bold;" + 
"		text-transform: uppercase;" + 
"" + 
"		background-color: rgba(8,122,191,0.98);" + 
"		color: #fff !important;" + 
"		opacity: 1;" + 
"" + 
"		-webkit-animation: fade 0.25s;" + 
"		animation: fade 0.25s;" + 
"	}" + 
"" + 
"	@-webkit-keyframes fade {" + 
"	    from {opacity: 0}" + 
"	    to {opacity: 1}" + 
"	}" + 
"	@keyframes fade {" + 
"	    from {opacity: 0}" + 
"	    to {opacity: 1}" + 
"	}	" + 
"" + 
"	#header_content ul li:hover > ul li {" + 
"" + 
"		padding: 10px;" + 
"		padding-top: 0px;" + 
"		padding-left: 20px;" + 
"		display: block;" + 
"		opacity:1;" + 
"	}" + 
"" + 
"	#header_content ul li:hover > ul a {" + 
"" + 
"		display: block;" + 
"		opacity:1;" + 
"		white-space:nowrap;" + 
"	}" + 
"" + 
"	#portal_menu_start_list {" + 
"" + 
"		left: 81px;" + 
"" + 
"		/* IE sauce */" + 
"		left:85px\\9;" + 
"" + 
"		/* Safari sauce */" + 
"		(;left:79px;);" + 
"	}" + 
"" + 
"	#portal_menu_atlases_list {" + 
"" + 
"		left: 290px;" + 
"		width: 210px !important;" + 
"" + 
"		padding-bottom:5px !important;" + 
"" + 
"		/* IE sauce */" + 
"		left: 303px\\9;	" + 
"" + 
"		/* Safari sauce */" + 
"		(;left:279px;);" + 
"	}" + 
"" + 
"	#portal_menu_tools_list {" + 
"" + 
"		left: 500px;" + 
"		width: 172px !important;" + 
"		height: 236px;" + 
"" + 
"		padding-bottom:5px !important;" + 
"" + 
"		/* IE sauce */" + 
"		left: 513px\\9;" + 
"		height:225px\\9;" + 
"" + 
"		/* Safari sauce */" + 
"		(;left:489px;);" + 
"		(;height:227px;);" + 
"	}" + 
"" + 
"	/* mozilla sauce */" + 
"	@-moz-document url-prefix() {" + 
"" + 
"		#header_content {" + 
"" + 
"			height:107px;" + 
"		}" + 
"" + 
"		#portal_menu_start_list {" + 
"" + 
"			left:85px;" + 
"		}" + 
"" + 
"		#portal_menu_atlases_list {" + 
"" + 
"			left: 302px;		" + 
"		}" + 
"" + 
"		#portal_menu_tools_list {" + 
"" + 
"			left: 512px;" + 
"			height:236px;" + 
"		}" + 
"	}" + 
"" + 
"	.portal_menu_list_heading {" + 
"" + 
"		color:#9ccae5;" + 
"		border-bottom:1px solid #9ccae5;" + 
"		padding-bottom:5px !important;" + 
"	}" + 
"" + 
"	.portal_menu_list_heading + li {" + 
"" + 
"		padding-top:10px !important;" + 
"	}" + 
"" + 
"	.portal_menu_list_heading li {" + 
"" + 
"		padding-bottom:0px;" + 
"	}	" + 
"" + 
"	#portal_menu_atlases_list li:last-of-type {" + 
"" + 
"		padding-bottom:15px !important;" + 
"	}" + 
"" + 
"	.icon-spin5 {" + 
"" + 
"		color:#000;" + 
"	}" + 
"" + 
"" + 
"</style>" + 
"<div id=\"pHeader\">" + 
"<div id=\"top_chrome\">" + 
"	<div id=\"top_chrome_buttons\">" + 
"		<div class=\"chrome_link_block\"><a href=\"http://www.alleninstitute.org\" target=\"_blank\" class=\"chrome_button\">ALLEN INSTITUTE</a></div>" + 
"		<div class=\"chrome_link_block chrome_active\"><a class=\"chrome_button\">BRAIN ATLAS</a></div>" + 
"	</div>" + 
"</div>" + 
"" + 
"<div id=\"header_content\">" + 
"" + 
"	<img id=\"portal_logo\" src=\"http://ibs-timd-ux2/sites/assets/images/portal_logo.png\"/>" + 
"" + 
"	<ul id=\"portal_tabs\" class=\"horizontal_list\">" + 
"		<li id=\"portal_tab_home\" class=\"portal_tab_item\"><a id=\"pHome\">HOME</a></li>" + 
"		<li id=\"portal_tab_start\" class=\"portal_tab_item\"><a id=\"pAnnouncements\">GET STARTED</a>" + 
"			<ul id=\"portal_menu_start_list\">" + 
"				<li><a id=\"pTutorials\" href=\"#\">Tutorials</a></li>" + 
"				<li><a id=\"pHighlights\" href=\"#\">Data Highlights</a></li>" +
"				<li><a id=\"pCaseStudies\" href=\"#\">Case Study</a></li>" +
"			</ul>" +
"		</li>" + 
"		<li id=\"portal_tab_help\" class=\"portal_tab_item\"><a id=\"pHelp\" target=\"_help_window\">HELP</a></li>" + 
"		<li id=\"portal_tab_menu\" class=\"portal_tab_item\"><a id=\"pMenuTitle\">Data & Tools</a><i class=\"icon-down-open icon_menu_down\"></i>" + 
"			<ul id=\"portal_menu_atlases_list\">" + 
"				<li class=\"portal_menu_list_heading\">Atlases</li>" + 
"				<li><a id=\"pMouseBrain\" href=\"#\">Mouse Brain</a></li>" + 
"				<li><a id=\"pDevelopingMouseBrain\" href=\"#\">Developing Mouse Brain</a></li>" + 
"				<li><a id=\"pHuman\" href=\"#\">Human Brain</a></li>" + 
"				<li><a id=\"pDevelopingHumanBrain\" href=\"#\">Developing Human Brain</a></li>" + 
"				<li><a id=\"pMouseConnectivity\" href=\"#\">Mouse Connectivity</a></li>" + 
"				<li><a id=\"pNonHumanPrimate\" href=\"#\">Non-Human Primate</a></li>" + 
"				<li><a id=\"pMouseSpinalCord\" href=\"#\">Mouse Spinal Cord</a></li>" + 
"				<li><a id=\"pGlioblastoma\" href=\"#\">Glioblastoma</a></li>" + 
"			</ul>" + 
"			<ul id=\"portal_menu_tools_list\">" + 
"				<li class=\"portal_menu_list_heading\">Other Tools</li>" + 
"				<li><a id=\"pAtlas\" href=\"#\">Reference Atlases</a></li>" + 
"				<li><a id=\"pAPI\" href=\"#\">API</a></li>" +
"			</ul>" +
"		</li>" + 
"	</ul>" + 
"" + 
"	<div id=\"portal_search_input_group\">" + 
"		<input type=\"text\" id=\"pSiteSearchTextInput\" tabIndex=\"0\" value=\"\" placeholder=\"Search...\"/>" + 
"		<i id=\"portal_search_button\" class=\"icon-search\"></i>" + 
"	</div>" + 
"" + 
"</div>" + 
"</div>" + 
 ""; 
    document.write(header_content);
}

function set_links(link_set, link_set_targets) {

	for(var id in link_set) {

		var a = document.getElementById(id);
		if(a) {
            a.setAttribute('href', link_set[id]);
            a.setAttribute('target', link_set_targets[id]);
        }
	}
}

// so far only the search widget events
function setup_events() {

    // _pSiteSearchTextInput is defined in portal.js
    var input = document.getElementById(_pSiteSearchTextInput);
    if(input && input.addEventListener) {

        input.addEventListener(
            "focus", 
            function() {

                this.value = "";                
            }, 
            false
        );
        input.addEventListener(
            "keyup", 
            function(event) {

                if(event.keyCode == 13)
                    do_site_search();    
            }, 
            false
        );
        document.getElementById('portal_search_button').addEventListener(
            "click", 
            function() {

                do_site_search();    
            }, 
            false
        );        
    } 
    // IE < 9 doesn't support addEventListener
    else if(input) {

        input.attachEvent(
            'onfocus', 
            function() {

                this.value = "";      
            }
        );
        input.attachEvent(
            "onkeyup", 
            function(event) {

                if(event.keyCode == 13)
                    do_site_search();    
            }
        );
        document.getElementById('portal_search_button').attachEvent(
            "onclick", 
            function() {

                do_site_search();    
            }
        );        
    }  
}

function do_site_search() {
    
    // set an indicator that something is happening
    document.getElementById('portal_search_button').className = "icon-spin5 animate-spin";

    // This method is defined in portal.js and will extract the
    // value from the text input.
    doSiteSearch();  
}

// _pEXTERNAL_ASSETS is defined in portal.js
import_portal();
// pTabLinks, _pTabLinkTargets is defined in portal.js
set_links(_pTabLinks, _pTabLinkTargets);
// so far just adding handlers to the site search input element
setup_events();

