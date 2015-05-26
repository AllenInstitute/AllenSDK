
function import_portal()  {

    var content = "" + 
"<style>" + 
"/* some styles used here are defined in and shared with portalHeader.html */" + 
"" + 
"#bottom_chrome {" + 
"" + 
"	position:relative;" + 
"	bottom:0px;" + 
"	width:100%;" + 
"	height:19px;" + 
"	padding-top:7px;" + 
"	background-color:#163E65;	" + 
"}" + 
"" + 
"#portal_copyright_notice {" + 
"" + 
"	font-family:arial,sans-serif;" + 
"	font-size:12px;" + 
"	color:#bbbcbd;" + 
"	margin:auto;" + 
"	text-align:center;" + 
"	position:relative;" + 
"}" + 
"" + 
"#footer_content {" + 
"" + 
"	position:relative;" + 
"	bottom:0px;" + 
"	width:calc(100% - 112px);" + 
"	height:65px;" + 
"	border-top:2px solid #A5A8AB;" + 
"	margin-left:50px;" + 
"	margin-right:50px;" + 
"}" + 
"" + 
"#portal_footer_item_container {" + 
"" + 
"	position:relative;" + 
"	top:16.5px;" + 
"	" + 
"	width:100%;" + 
"	height:25px;" + 
"	border:0px solid #000;" + 
"" + 
"	display:table;" + 
"}" + 
"" + 
".portal_footer_item {" + 
"" + 
"	font-family:arial,sans-serif;" + 
"	font-weight:bold;" + 
"	font-size:12px;" + 
"	cursor:pointer;" + 
"	display:inline-block;" + 
"	border: 0px solid #f00;" + 
"	text-align:center;" + 
"	vertical-align:middle;" + 
"	width:20%;" + 
"	height:25px;" + 
"}" + 
"" + 
".portal_social_icon_set {" + 
"" + 
"	font-size:20px;" + 
"	color:#fff;	" + 
"}" + 
"" + 
".portal_social_icon_set i {" + 
"" + 
"	background-color:#A5A8AB;	" + 
"}" + 
"" + 
".portal_call_to_action {" + 
"" + 
"	background-color: #F9E200;" + 
"	color: #1c598d;" + 
"" + 
"	font-size:14px;" + 
"	height:20px;" + 
"	width:180px;" + 
"	padding-top:4px;" + 
"}" + 
"" + 
".portal_call_to_action:hover {" + 
"" + 
"	background-color: #0779be;" + 
"	color: #fde500" + 
"}" + 
"" + 
"#portal_citation_policy {" + 
"" + 
"	float:right;" + 
"}" + 
"" + 
".portal_link_stack a {" + 
"" + 
"	color:#0779BE !important;" + 
"}" + 
"" + 
"/* overriding application tab bar styles */" + 
"#pTabBar {" + 
"" + 
"	width:calc(100% - 100px) !important;" + 
"	margin:auto;" + 
"	background-color:#fff;	" + 
"	padding:10px;" + 
"	margin-bottom: 15px;" + 
"" + 
"	/* IE Sauce */" + 
"	width:auto\\9;" + 
"	margin-left:50px\\9;	" + 
"	margin-right:50px\\9;" + 
"}" + 
"" + 
"#pTabBar td {" + 
"" + 
"	height:32px;" + 
"	padding-left:10px;" + 
"	padding-right:0px;" + 
"}" + 
"" + 
"#pTabBar a {" + 
"" + 
"	color: #0779BE;" + 
"	text-decoration: none;" + 
"	background-color: #fff;" + 
"	padding-left: 0px;" + 
"	padding-right: 30px;" + 
"	text-align: center;" + 
"	border: none !important;" + 
"	text-transform:uppercase;" + 
"	font-family:Arial,Sans-Serif;" + 
"	font-size:12px;" + 
"	font-weight:bold;" + 
"}" + 
"" + 
"#pTabBar a:hover {" + 
"" + 
"	color: #0779BE;" + 
"	background-color: #fff;" + 
"}" + 
"" + 
"#pTabBar a.pTabBarSelected {" + 
"" + 
"	color: #A5A8AB;" + 
"	border: solid 0px #fff;" + 
"}" + 
"" + 
"#top_menu {" + 
"" + 
"	background-color:#0779be;" + 
"	color:#fbe300;" + 
"}" + 
"" + 
"#app_menu_table {" + 
"" + 
"	border-spacing:0px;" + 
"}" + 
"" + 
"#menu_cell {" + 
"" + 
"	color:#0779be !important;" + 
"}" + 
"" + 
"#menu_cell .pTabBarSelected {" + 
"" + 
"	color:#a5a8ab !important;" + 
"}" + 
"" + 
"#menu_cell:hover {" + 
"" + 
"	background-color:#0779be !important;" + 
"	color:#fbe300 !important;" + 
"}" + 
"" + 
"#menu_cell:hover > ul ul {" + 
"" + 
"	display:block;" + 
"	-webkit-animation: fade 0.25s;" + 
"	animation: fade 0.25s;	" + 
"}" + 
"" + 
"#app_menu_table li {" + 
"" + 
"	background-color:inherit !important;" + 
"	color:inherit !important;" + 
"}" + 
"" + 
"#app_menu_table ul a {" + 
"" + 
"	background-color:inherit !important;" + 
"	color:inherit !important;" + 
"}" + 
"" + 
"#app_menu_table ul a:hover {" + 
"" + 
"	color:#fbe300 !important;" + 
"}" + 
"" + 
"#nav_menus {" + 
"" + 
"	background-color: #0779be;" + 
"	color: #fff;" + 
"	top: 23px;	" + 
"	min-width:156px !important;" + 
"}" + 
"" + 
"#top_menu img {" + 
"" + 
"	display:none;" + 
"}" + 
"" + 
"" + 
"#nav ul {" + 
"" + 
"	left:-11px !important;" + 
"}" + 
"" + 
"#nav ul li {" + 
"" + 
"	border-top:none;" + 
"}" + 
"" + 
"#nav ul a {" + 
"" + 
"	text-align: left;" + 
"	padding-left: 14px;	" + 
"}" + 
"" + 
"#nav ul a:hover {" + 
"" + 
"	background-color:#0779be;" + 
"	color:#fbe300;" + 
"}" + 
"" + 
"#nav li {" + 
"" + 
"	padding-right:10px !important;" + 
"}" + 
"" + 
"#nav li a {" + 
"" + 
"	color:#0779BE;	" + 
"}" + 
"" + 
".portal_icon {" + 
"" + 
"	color:#fff !important;" + 
"}" + 
"" + 
".icon-twitter:hover {" + 
"" + 
"	background-color:#5fa8dc;" + 
"}" + 
".icon-facebook:hover {" + 
"" + 
"	background-color:#3d5a98;" + 
"}" + 
".icon-youtube-play:hover {" + 
"" + 
"	background: linear-gradient(#e6312a, #b42227);" + 
" background-color: #CB2029\0; /* IE9 only */" +
"}" + 
".icon-linkedin-3:hover {" + 
"" + 
"	background-color:#007bb5;" + 
"}" + 
"" + 
"/* This is a global change to link colors that will affect all the apps */" + 
"a:hover {" + 
"" + 
"	color:#47AEDC !important;" + 
"}" + 
"" + 
"/* app-specific overrides */" + 
"#search_result_common_container {" + 
"" + 
"	width: calc(100% - 80px);" + 
"	margin-left: 20px;" + 
"}" + 
"" + 
"html, body {" + 
"" + 
"	height:100%;" + 
"}" + 
"" + 
".siteContent, .pageContent {" + 
"" + 
"	min-height: calc(100% - 125px) !important;" + 
"}" + 
"" + 
".separator {" + 
"" + 
"	height:0px;" + 
"}" + 
"" + 
"#preview_cell {" + 
"" + 
"	min-width:251px !important;" + 
"}" + 
"" + 
"" + 
"/* changing the header bar colors to match the top chrome color */" + 
"th.contentBlock, .ui-widget-header, ._lw_header, .ageaTabSelected, " + 
"._cs_header_text_cell, .sectionHeader, .section_header, " + 
".projection_search_widget_header, .container_header, .aboutHeader {" + 
"" + 
"	background-color:#163e65 !important;" + 
"}" + 
"" + 
"/* meta-differential search result table headers */" + 
".metaSearchResultTable th {" + 
"" + 
"	background-color:#163e65 !important;" + 
"}" + 
".metaSearchResultTable th.subheader {" + 
"" + 
"	background-color:#fff !important;" + 
"}" + 
"" + 
"/* projection landing page headers */" + 
"table.projection_search_widget th {" + 
"" + 
"	background-color:#163e65 !important;	" + 
"}" + 
"" + 
"/* connectivity experiment page (thumbnails) top widget header */" + 
"._experiment_specimen_widget tr:first-of-type td {" + 
"" + 
"	background-color:#163e65 !important;		" + 
"}" + 
"" + 
"/* AGEA about box header */" + 
".aboutAgeaOverlay div:first-child {" + 
"" + 
"	background-color:#163e65 !important;	" + 
"}" + 
"" + 
"/* AGEA in devmouse */" + 
".pageContentAgea {" + 
"" + 
"	margin-bottom: 52px !important;" + 
"}" + 
"" + 
"/* AGEA in ABA */" + 
".pageContent .pageContentAgea {" + 
"" + 
"	margin-bottom: 16px !important;" + 
"}" + 
"" + 
"/* some suggestions for app content..." + 
"" + 
".slick-header-column, .ui-state-default {" + 
"" + 
"	background:none !important;" + 
"	background-color:#eee !important;" + 
"}" + 
"" + 
"#search_container {" + 
"" + 
"	width:calc(100% - 100px);" + 
"	margin:auto;" + 
"	margin-left:auto;" + 
"	margin-right:auto;" + 
"	margin-bottom:12px;" + 
"	border:1px solid #bbb;" + 
"}" + 
"" + 
"#result_block, #preview_block, #search_container {" + 
"	" + 
"	-webkit-border-radius: 0px;" + 
"	-moz-border-radius: 0px;" + 
"	border-radius: 0px;	" + 
"}" + 
"" + 
"#result_block:hover, #preview_block:hover, #search_container:hover {" + 
"" + 
"	-webkit-box-shadow: 1px 1px 4px 0px rgba(0,0,0,0.15);" + 
"	-moz-box-shadow: 1px 1px 4px 0px rgba(0,0,0,0.15);" + 
"	box-shadow: 1px 1px 4px 0px rgba(0,0,0,0.15);" + 
"}" + 
"*/" + 
"" + 
"</style>" + 
"" + 
"" + 
"<div id=\"pFooter\">" + 
"<div id=\"footer_content\">" + 
"" + 
"	<div id=\"portal_footer_item_container\">" + 
"" + 
"		<a id=\"pContactUs\" target=\"_blank\"><span id=\"portal_send_message\" class=\"portal_footer_item portal_call_to_action\">SEND US A MESSAGE <i class=\"icon-right-open\"></i></span></a>" + 
"		<span class=\"portal_footer_item portal_social_icon_set\">" + 
"			<a id=\"pTwitter\" target=\"_blank\"><i class=\"icon-twitter portal_icon\"></i></a>" + 
"			<a id=\"pFacebook\" target=\"_blank\"><i class=\"icon-facebook portal_icon\"></i></a>" + 
"			<a id=\"pYouTube\" target=\"_blank\"><i class=\"icon-youtube-play portal_icon\"></i></a>" + 
"			<a id=\"pLinkedIn\" target=\"_blank\"><i class=\"icon-linkedin-3 portal_icon\"></i></a>" + 
"		</span>" + 
"		<span class=\"portal_footer_item portal_link_stack\">" + 
"			<a id=\"pAbout\" target=\"_blank\">About the Allen Institute</a></a>" + 
"			<br/>" + 
"			<a id=\"pPublications\" target=\"_blank\">Allen Institute Publications</a></a>" + 
"		</span>		" + 
"		<span class=\"portal_footer_item portal_link_stack\">" + 
"			<a id=\"pPrivacyPolicy\" target=\"_blank\">Privacy Policy</a></a>" + 
"			<br/>" + 
"			<a id=\"pTermsOfUse\" target=\"_blank\">Software License</a></a>" + 
"		</span>		" + 
"		<a id=\"pCitationPolicy\" target=\"_blank\"><span id=\"portal_citation_policy\" class=\"portal_footer_item portal_call_to_action\">CITATION POLICY <i class=\"icon-right-open\"></i></span></a>" + 
"	</div>" + 
"</div>" + 
"" + 
"" + 
"<div id=\"bottom_chrome\">" + 
"	<div id=\"portal_copyright_notice\">Copyright ©2015. Allen Institute for Brain Science. All Rights Reserved.</div>" + 
"</div>" + 
"</div>" + 
 ""; 
    document.write(content);
}

/* 
    Sets the href attribute for all the defined pFooterLinks keys.
    This is done programatically because the links must be absolute, 
    and the base url is configured differently in portal.js for each 
    environment.
*/    
function set_links(link_set) {

	for(var id in link_set) {

		var a = document.getElementById(id);
		if(a)
			a.setAttribute('href', link_set[id]);
	}
}

/*
    Sometimes it's easier to inject updates from here rather than try to
    rework the legacy apps individually.
*/
function hack_legacy_app_specific_hacks() {

    if(_pTabId == "pMouseSpinalCord") {

        if(window.console)
            console.log("importing legacy spinal hacks");

        import_spinal_hacks();

    } else if(_pTabId == "pGlioblastoma") {

        import_glio_hacks();
    }
}

function import_spinal_hacks() {

    var content = "<style>" + 
"	/* " + 
"		Style overrides to get the spinal cord app to fit in a " + 
"		bit better with the changes done for the June 2014 portal redesign." + 
"	*/" + 
"" + 
"" + 
"	/* getting the footer to sit at the foot of the page */" + 
"    .siteContent {" + 
"" + 
"    	min-height:calc(100% - 93px) !important;" + 
"    }    " + 
"" + 
"    /* search area background and controls */" + 
"    #searchArea {" + 
"" + 
"    	background-color:#eee;" + 
"    }" + 
"" + 
"	#searchControls {" + 
"" + 
"		width: calc(100% - 100px);" + 
"		margin: auto;" + 
"	}" + 
"" + 
"    input.btnSearch {" + 
"" + 
"    	color:inherit !important;" + 
"    	background-color: buttonface !important;" + 
"    }" + 
"" + 
"    input.submit {" + 
"" + 
"        color:inherit !important;" + 
"        background-color: buttonface !important;" + 
"    }    " + 
"</style>" + 
 ""; 
    document.write(content);    
}

function import_glio_hacks() {

    var content = "<style>" + 
"	/* " + 
"		Style overrides to get the spinal cord app to fit in a " + 
"		bit better with the changes done for the June 2014 portal redesign." + 
"	*/" + 
"" + 
"" + 
"	/* getting the footer to sit at the foot of the page */" + 
"    .siteContent {" + 
"" + 
"    	min-height:calc(100% - 93px) !important;" + 
"    }    " + 
"" + 
"    /* search area background and controls */" + 
"    #searchArea {" + 
"" + 
"    	background-color:#eee;" + 
"    }" + 
"" + 
"	#searchControls {" + 
"" + 
"		width: calc(100% - 100px);" + 
"		margin: auto;" + 
"	}" + 
"" + 
"    input.btnSearch {" + 
"" + 
"    	color:inherit !important;" + 
"    	background-color: buttonface !important;" + 
"    }" + 
"" + 
"    input.submit {" + 
"" + 
"        color:inherit !important;" + 
"        background-color: buttonface !important;" + 
"    }    " + 
"</style>" + 
"" + 
 ""; 
    document.write(content);    
}


// _pEXTERNAL_ASSETS is defined in portal.js
import_portal();

// _pFooterLinks is defined in portal.js. This programatically 
// updates href attributes in the footer.
set_links(_pFooterLinks);

// hackage specific to the legacy apps that we're not otherwise touching.
hack_legacy_app_specific_hacks();

