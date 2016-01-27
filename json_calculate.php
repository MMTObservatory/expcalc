

<?php
  ini_set('display_errors', 'On');
  $params = array();

  $params["spectrograph"] = $_REQUEST["instrument"];
  $params["outputType"] = $_REQUEST["outputType"];

  //Check for the output type mode and get the "crtical value"
  if ($params["outputType"] == "1"){
    $params["depth"] = floatval($_REQUEST["sn"]);
    $params["calcType"] = 'Time';
  } else {
    $params["depth"] = floatval($_REQUEST["time"]);
    $params["calcType"] = 'SNR';
  };

  //Grab the other parameters we need
  $params["grating"] = $_REQUEST["grating_selector"];
  $params["order"] = intval($_REQUEST["order"]);
  $params["cenwave"] = floatval($_REQUEST["cenwave"]);
  $params["filter"] = $_REQUEST["filters"];
  $params["binning_spatial"] = intval($_REQUEST["spatialBinning"]);
  $params["binning_spectral"] = intval($_REQUEST["spectralBinning"]);
  $params["slit_width"] = floatval($_REQUEST["slitplates"]);
  $params["seeing"] = floatval($_REQUEST["seeing"]);
  $params["specmag"] = floatval($_REQUEST["ABmag"]);
  $params["lunar_phase"] = floatval($_REQUEST["lunarphase"]);
  $params["airmass"] = floatval($_REQUEST["airmass"]);


$errors = array(); // To Store Errors
$form_data = array(); // Pass back to index

//Execute the python script with JSON data

$result = shell_exec("python python/json_exptime.py " . escapeshellarg(json_encode($params)));

//Decode the result
$form_data = json_decode($result, true);
$form_data['command'] = "python python/json_exptime.py " . escapeshellarg(json_encode($params));

if (!empty($errors)) { // If there were any errors
  $form_data['success'] = false;
  $form_data['errors'] = $errors;
} else { // If no, process the form
  $form_data['success'] = true;
}

echo json_encode($form_data);

 ?>
