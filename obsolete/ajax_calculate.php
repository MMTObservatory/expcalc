

<?php

  $params = array();

  $params["instrument"] = $_REQUEST["instrument"];
  $params["outputType"] = $_REQUEST["outputType"];

  //Check for the output type mode and get the "crtical value"
  if ($params["outputType"] == "1"){
    $params["depth"] = $_REQUEST["sn"];
    $params["calcType"] = 'Time';
  } else {
    $params["depth"] = $_REQUEST["time"];
    $params["calcType"] = 'SNR';
  };

  //Grab the other parameters we need
  $params["grating"] = $_REQUEST["grating_selector"];
  $params["order"] = $_REQUEST["order"];
  $params["cenwave"] = $_REQUEST["cenwave"];
  $params["filters"] = $_REQUEST["filters"];
  $params["spatialBinning"] = $_REQUEST["spatialBinning"];
  $params["spectralBinning"] = $_REQUEST["spectralBinning"];
  $params["slit"] = $_REQUEST["slitplates"];
  $params["seeing"] = $_REQUEST["seeing"];
  $params["ABmag"] = $_REQUEST["ABmag"];
  $params["lunarphase"] = $_REQUEST["lunarphase"];
  $params["airmass"] = $_REQUEST["airmass"];


$errors = array(); // To Store Errors
$form_data = array(); // Pass back to index

$script = "./exptime.py";
$command = escapeshellcmd($script . " " .
  $params["instrument"] . " " .
  $params["outputType"] . " " .
  $params["depth"]. " " .
  $params["grating"] . " " .
  $params["order"] . " " .
  $params["cenwave"] . " " .
  $params["filters"] . " " .
  $params["spatialBinning"] . " " .
  $params["spectralBinning"] . " " .
  $params["slit"] . " " .
  $params["seeing"] . " " .
  $params["ABmag"] . " " .
  $params["lunarphase"] . " " .
  $params["airmass"]);

$output = shell_exec($command);
var_dump($output);
$out = json_decode($output, true);

if (!empty($errors)) { // If there were any errors
  $form_data['success'] = false;
  $form_data['errors'] = $errors;
} else { // If no, process the form
  $form_data['success'] = true;
  $form_data['output'] = $command;
  $form_data['other'] = $out;

}

echo json_encode($form_data);

 ?>
