<!doctype html>
<html lang="us">

<head>
	<meta charset="utf-8">
	<title>MMTO f/9 Spectrograph Exposure Time Calculator</title>
	<link href=" css/BlackTie/jquery-ui.css" rel="stylesheet">
	<link href="css/main.css" rel="stylesheet" type="text/css">
	<script src="css/BlackTie/external/jquery/jquery.js"></script>
	<script src="http://code.jquery.com/jquery-1.9.1.js" type="text/javascript"></script>
	<script type='text/javascript' src='http://ajax.googleapis.com/ajax/libs/jqueryui/1.9.1/jquery-ui.js'></script>
	<!--This is from the css example and includes some useful scripts. I moved it to js/ for cleanliness -->
	<script src="js/expcalc.js"></script>
	<script src="http://code.highcharts.com/highcharts.js" type="text/javascript"></script>







</head>

<body>

	<h2> MMTO f/9 Spectrograph Exposure Time Calculator </h2>
	<div id="container">

		<div id="left_container">
			<!-- Initialize the form. This will need an onSubmit eventually -->
			<form id="exptime_form" method="post" action="index.php">
				<div id="accordion">

					<h3 id="accordion_step1">1. Select Instrument</h3>
					<div>
						<div class="formContainer">
							<div>
								<!-- Spectrograph selection drop down. -->
								<select name="instrument" onChange="onInstrumentChange(value)" id="instrument">
									<option value="none">Select Spectrograph</option>
									<option value="blue">Blue Channel</option>
									<option value="red">Red Channel</option>
								</select>
							</div>
							<div>
								<button type="button" value="next" onClick="selectInstrument();">Next</button>
							</div>
						</div>
					</div>

					<h3 id="accordion_step2">2. Calculation Type</h3>
					<div class="formContainer">
						What would you like to calculate?
						<select id="outputType" name="outputType">
							<option value="1">Integration Time</option>
							<option value="2">Signal-To-Noise</option>
						</select>
						<button type="button" value="next" onClick="selectOutput();">Next</button>
					</div>


					<h3 id="accordion_step3">3. Observation Parameters</h3>

					<div class="formContainer">

						<!-- Boiler plate in case the user gets to tab 2
						without selecting output type -->
						<div id="outputType0">
							<span class='bold' style="color:red">Please Choose <u>Calculation Type</u> before continuing.
							</span>
						</div>

						<!-- Enter the desired exposure time. This should only be visible if
						the user selected this as the input -->
						<div id="outputType1">
							<span class="bold">Integration Time:</span>
							<span class="parameterinput">
								<input name="time" onFocus="focusIntegrationTime();" type="text" id="time" value="" size="10" maxlength="6" />
							</span>seconds
						</div>

						<!-- Enter the desired Signal/Noise Ratio.  This should only be
						visible if the user selected this as the input -->
						<div id="outputType2">
							<span class="bold"> Signal-to-Noise: </span>
							<span class="parameterinput">
								<input name="sn" onFocus="focusSignalNoise();" type="text" id="sn" value="1.2" size="6" maxlength="6" />
							</span>
						</div>

						<!-- Create the form to hold the grating drop down. The JS code
					handles filling this in based on the instrument selected -->
						<div>
							<span class="bold"> Grating: </span>
							<div class="subparam">Available Gratings:
								<select onFocus="focusGratings();" name="grating_selector" id="grating_selector" disabled>
									<option value="noinstrument" selected> Select Instrument</option>
								</select>
							</div>
						</div>

						<!-- Grating order -->
						<div class="subparam">
							Order:
							<input name="order" type="text" id="order" value="1" size="5" maxlegenth="1" />
						</div>

						<!--Central Wavelength -->
						<div class="subparam">
							Central Wavelength:
							<input onFocus="focusCentralWavelength();" name="cenwave" type="text" id="cenwave" value="" size='10' />
						</div>

						<!-- Filter May as well go here as it makes the most sense -->
						<div class="subparam">
							Filter:
							<select name="filters" id="filters" onFocus="focusFilters();">
								<option value="Clear" selected>Clear</option>
								<!-- Blue Blockers -->
								<option value="UV36">UV-36</option>
								<option value="L38">L-38</option>
								<option value="L42">L-42</option>
								<option value="LP594">LP-495 (Y-50)</option>
								<option value="LP530">LP-530</option>
								<option value="R63">R-63</option>
								<!--Red blockers -->
								<option value="CM500">CM500</option>
								<option value="CuSO4">CuSO4</option>
								<option value="C500">C-500</option>
								<option value="U330">U330</option>
								<!--Neutral Density -->
								<option value="05ND">0.5m Neutral Density</option>
								<option value="125ND">1.25m Neutral Density</option>
								<option value="25ND">2.5m Neutral Density</option>



							</select>
						</div>
						<!-- Binning : -->
						<div>
							<span class='bold'>Binning:</span>

							<!-- Spatial -->
							<div class="subparam">
								Spatial Binning:
								<input name="spatialBinning" type="text" id="spatialBinning" value="2" size="5" maxlegnth=2 onFocus="focusBinning();"/>
							</div>

							<!-- Spectral -->
							<div class="subparam">
								Dispersion Binning:
								<input name="spectralBinning" type="text" id="spectralBinning" value="1" size="5" maxlength=2 onFocus="focusBinning();"/>
							</div>

						</div>

						<!-- Slit -->
						<div>
							<span class='bold'>Slit Width: </span>
							<select name="slitplates" id="slitplates" onFocus="focusAperture();">
								<option value="noslit" selected>Select Slit Width:</option>
								<option value="0.75">0.75"</option>
								<option value="1.00">1.00"</option>
								<option value="1.25">1.25"</option>
								<option value="1.50">1.50"</option>
								<option value="2.00">2.00"</option>
								<option value="3.50">3.50"</option>
								<option value="5.00">5.00"</option>
							</select>
						</div>

						<!-- AB Magnitude -->
						<!-- For now, the software assumes a flat-spectrum AB magnitude
						source for the calculation. We can change this later, but not yet -->
						<div>
							<span class='bold'>AB Magnitude: </span>
							<input onFocus="focusABMagnitude();" name="ABmag" id="ABmag" type='text' size=4 />
						</div>

						<div>
							<!-- Make a sub container for the observing conditons -->
							<span class="bold"> Observing Conditions: </span>

							<!-- Seeing first -->
							<div class="subparam">
								Seeing:
								<input name='seeing' type="text" id="seeing" value="1.0" size="5" maxlegnth=5 onFocus="focusSeeing();"/> arcsec
							</div>

							<!-- Lunar phase -->
							<div class="subparam">
								Lunar Phase:
								<input name='lunarphase' type='text' id='lunarphase' value='0' size='5' maxlength=2 onFocus="focusLunarPhase();"/>
							</div>

							<!-- Airmass -->
							<div class="subparam">
								Airmass:
								<input name='airmass' type='text' id='airmass' value="1.0" size="5" maxlegnth=4 onFocus="focusAirmass();"/>
							</div>

							<!-- The magic "GO" Button -->
							<div>
								<button id="submit" type="button" onClick="Run();"> Calculate </button>
							</div>



						</div>
					</div>
				</div>
			</form>
		</div>

		<div id="right_container">
			<div id="result_container">
				<div id="tabs">
					<ul>
						<li><a href="#tabs-1">General Information</a></li>
						<li><a href="#tabs-2">Results</a></li>
						<li><a href="#tabs-3">Help</a></li>
					</ul>
					<div id="tabs-1" class="floater_help">
						<iframe name="info_frame" id="info_frame" src="info.html" width="100%" height="400" frameborder="0"></iframe>
					</div>
					<div id="tabs-2">
						<div id="loading">
							<img src="images/ajax-loader.gif" width="16" height="16"> Calculating </div>
						<span id="summary">
							<div class="resultparam" id="snrresult">
								Signal-to-Noise:
								<input name="finalsnr" type="text" id="finalsnr" value="" size=20 maxlength=10 disabled align="right" />
							</div>
							<div class="resultparam" id="timeresult">
								Exposure Time:
								<input name="finalexptime" type="text" id="finalexptime" value="" size=20 maxlength=10 disabled align="right" /> Seconds
							</div>
							<div class="resultbuttons">
								<button id="showCountsPlot" type="button" onClick="ShowCountsPlot();">Show Plot of Counts </button>
								<button id="showSNRPlot" type="button" onClick="ShowSNRPlot();"> Show Expected SNR Plot </button>


							</div>

						</span>
						<div id="plotcontainer"></div>
						<div id="snrcontainer"></div>



					</div>
					<div id="tabs-3">
					</div>
				</div>
			</div>

		</div>


		<script language="javascript">
			// The script that does the bulk of the work.
			function Run() {

				//Run the validation script to ensure that we have everything we need
				if (validate()) {
					//Show the Summary tabs
					$("#tabs").tabs("select", 1); // Select "Results tab"
					$("#loading").show();

					$("#showSNRPlot").hide();
					$("#showCountsPlot").hide();
					$("#timeresult").hide();
					$("#snrresult").hide();
					$("#plotcontainer").hide();
					$("#snrcontainer").hide();

					//Scroll to top of the page.
					$('html, body').animate({
						scrollTop: 0
					}, 'slow');


					//Serialize the settings and pass to JSON
					var params = $("#exptime_form").serialize();
					var url = "json_calculate.php";
					$.ajax({
						type: "POST",
						url: url,
						data: params, //send our JSON string as data
						dataType: "json",
						success: OnResult,
						error: OnError
					});
				}

			}


			function ShowCountsPlot() {
				$("#plotcontainer").show();
				$("#snrcontainer").hide();
				$("#timecontainer").hide();

				$("#showCountsPlot").hide();
				$("#showSNRPlot").show();
			}

			function ShowSNRPlot() {
				$("#snrcontainer").show();
				$("#plotcontainer").hide();

				$("#showCountsPlot").show();
				$("#showSNRPlot").hide();
			}




			function OnError(data) {
				console.log(data);
			//	$("#summary").html("Error: There was a problem");
				$("#loading").hide();
				$("#plotcontainer").hide();
				$("#snrcontainer").hide();
			}

			//If we succeed
			function OnResult(data) {
				console.log(data);


				var exptime_in = data.exptime_median;
				var snr_in = data.snr_median;
				var exptime = exptime_in.toFixed(1);
				var snr = snr_in.toFixed(2);



				$("#loading").hide();



				$("#showSNRPlot").hide();
				$("#showCountsPlot").show();
				$("#timeresult").hide();
				$("#snrresult").hide();


				//Show the results
				if (data.calcType == "Time") {
					$("#timeresult").show();
					$("#finalexptime").val(exptime);

				}
				if (data.calcType == "SNR") {
					$("#snrresult").show();
					$("#finalsnr").val(snr);
				}


				$(function() {
					$('#plotcontainer').highcharts({
						chart: {
							defaultSeriesType: 'line',
							zoomType: 'x',
							marginRight: 40,
							marginLeft: 80,
						},
						credits: {
							enabled: false,
						},
						title: {
							text: 'Expected Counts',
							x: -20 //center
						},

						xAxis: {
							title: {
								text: "Wavelength (Angstroms)"
							},
							startOnTick: false,
							endOnTick: false,
							gridLineWidth: 1,
							showLastLabel: true
						},
						yAxis: {
							title: {
								text: 'Counts / Pixel'
							},
							plotLines: [{
								value: 0,
								width: 1,
								color: '#808080'
							}]
						},
						tooltip: {
							valueSuffix: ' counts'
						},
						legend: {
							layout: 'vertical',
							align: 'right',
							verticalAlign: 'top',

							borderWidth: 1,
							x: -80,
							y: 50,
						},
						series: [{
							name: 'Sky',
							data: data.skycounts,
							color: 'rgba(126,86,134,.9)',

						}, {
							name: 'Object',
							data: data.objcounts,

						}, {
							name: 'Total Counts',
							data: data.totalcounts,
							color: 'rgba(248,161,63,1)'
						}]
					});
				});


				$(function() {
					$('#snrcontainer').highcharts({
						chart: {
							defaultSeriesType: 'line',
							zoomType: 'x',
							marginRight: 20,
							marginLeft: 80,
						},
						credits: {
							enabled: false,
						},
						title: {
							text: 'Expected Signal to Noise',
							x: -20 //center
						},

						xAxis: {
							title: {
								text: "Wavelength (Angstroms)"
							},
							startOnTick: false,
							endOnTick: false,
							gridLineWidth: 1,
							showLastLabel: true
						},
						yAxis: {
							title: {
								text: 'Signal to Noise Ratio / pixel'
							},
							plotLines: [{
								value: 0,
								width: 1,
								color: '#808080'
							}]
						},
						tooltip: {
							formatter: function() {
								return ' ' + this.x + ' Ang, ' + this.y.toFixed(1) + ' S/N';
							}
						},
						legend: {
							layout: 'vertical',
							align: 'right',
							verticalAlign: 'top',
							x: -30,
							y: 100,
							borderWidth: 1
						},
						series: [{
							name: 'SNR',
							data: data.snr,
							color: 'rgba(223,83,83,0.5)',

						}]
					});
				});

				//Show this one by default
				$("#snrcontainer").show();
				$("#plotcontainer").hide();


			}

			//Check to make sure we have everything se need
			function validate() {
				clearErrors(); // clear any previous error messages
				var messages = Array();
				var success = true;
				var numberPattern = new RegExp("[0-9]"); // RegExp for number checking
				var letterPattern = new RegExp("[a-zA-Z ]"); //RegExp for letter checking

				//Check to ensure an exposure time was given (or the code set it to INDEF)
				if ($("#time").val() == "" || !numberPattern.test($("#time").val())) {
					// Only worry if outputType == 2 (calculate a SNR)
					if ($("#outputType").val() == 2) {
						if ($("#time").val() != "INDEF") {
							messages.push("Please enter 'Exposure Time'" + $("#time").val());
							$("#time").addClass("error");
							success = false;
						}
					}
				}

				// Check the SNR
				if (($("#sn").val() == "" || !numberPattern.test($("#sn").val()))) {
					//Only worry about this if outputtype == 1
					if ($("#outputType").val() == 1) {
						messages.push("Please enter Signal-to-Noise.\n");
						$("#sn").addClass("error");
						success = false;
					}
				}

				//Check the grating
				if ($("#grating_selector").val() == "gratingnone" || $("#grating_selector").val() == "noinstrument") {
					messages.push("Please select a Grating \n");
					$("#grating_selector").addClass("error");
					success = false;
				}

				//Check that the order was specified
				if ($("#order").val() == "" || !numberPattern.test($("#order").val())) {
					messages.push("Please entered the grating order");
					$("#order").addClass("error");
					success = false;
				}

				//Check the central wavelength
				if ($("#cenwave").val() == "" || !numberPattern.test($("#cenwave").val())) {
					messages.push("Please enter the desired central wavelength");
					$("#cenwave").addClass("error");
					success = false;
				}

				//Check the magnitude
				if ($("#ABmag").val() == "" || !numberPattern.test($("#ABmag").val())) {
					messages.push("Please specify a source magnitude");
					$("#ABmag").addClass("error");
					success = false;
				}

				//Check the seeing
				if ($("#seeing").val() == "" || !numberPattern.test($("#seeing").val())) {
					messages.push("Please enter a seeing");
					$("#seeing").addClass("error");
					success = false;
				}

				//Check the airmass
				if ($("#airmass").val() == "" || !numberPattern.test($("#airmass").val())) {
					messages.push("Please enter an airmass");
					$("#airmass").addClass("error");
					success = false;
				}

				//Check the lunar phase
				if ($("#lunarphase").val() == "" || !numberPattern.test($("#lunarphase").val())) {
					messages.push("Please enter a lunar phase");
					$("#lunarphase").addClass("error");
					success = false;
				}

				//Check the slit length
				if ($("#slitplate").val() == "noslit") {
					messages.push("Please select a slit");
					$("#slitplate").addClass("error");
					success = false;
				}

				//Check spatial binning
				if ($("#spatialBinning").val() == "" || !numberPattern.test($("#spatialBinning").val())) {
					messages.push("Please enter a valid spatial binning value");
					$("#spatialBinning").addClass("error");
					success = false;
				}

				//check spectral binning
				if ($("#spectralBinning").val() == "" || !numberPattern.test($("#spectralBinning").val())) {
					messages.push("Please enter a valid spectral binning value");
					$("#spectralBinning").addClass("error");
					success = false
				}

				if (!success) {
					alert(messages);
					//alert("Error: You are missing some required fields. Please enter values and retry");
					return false;
				}

				return true;
			}
		</script>



</body>

</html>
