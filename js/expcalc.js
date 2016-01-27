
$(document).ready(function() {

  $("#accordion").accordion({
    header: "h3",
    autoHeight: false
  });
  $("#accordion").bind('accordionchange', onAccordionChange);

  //We want these hidden until the user selects the method
  $("#outputType1").hide();
  $("#outputType2").hide();

  $(window).scroll(onScroll);
  $("#tabs").tabs();

  $("#loading").hide(); // Only want to show the loading bar if we hit the calculate button

});

function onScroll() {
  var scrollTop = $(document).scrollTop();

  $("#test_output").html(scrollTop);
  $("#floater").css('top', scrollTop);

  if (scrollTop > 60) {
    $("#tabs-4").css('top', scrollTop - 60);
  } else {
    $("#tabs-4").css('top', 0);
  }
}

function selectInstrument() {
  // This function will take the selected spectrograph, and trigger the next
  // accordion to open accordingly.
  var whichone = $('#instrument').val(); // Read the "instrument" <select> value
  selectedInstrument = whichone;
  $("#accordion").accordion("option", "active", 1);

}

function onInstrumentChange(value) {
  switch (value) {
    case "red":
      FillRedGratings();
      window.frames.info_frame.SelectRed();
      break;
    case "blue":
      FillBlueGratings();
      window.frames.info_frame.SelectBlue();
      break;
    default:
      ClearGratings();
      window.frames.info_frame.SelectNone();
      break;
  }
}

//Trigger this function when the user specifies which kind
//of output they would like
function selectOutput() {
  var whichoutput = $('#outputType').val();
  outputType = whichoutput;
  if (outputType == "1") {
    //Integration time was selected
    $("#time").val("INDEF");
    $("#sn").val("");
    $("#outputType0").hide();
    $("#outputType2").show();
    $("#outputType1").hide(); // Hide S/N box
  } else {
    //Signal to Noise Selected
    $("#sn").val("INDEF");
    $("#time").val("");
    $("#outputType0").hide();
    $("#outputType2").hide(); //Hide exptime box
    $("#outputType1").show();
  }
  $("#accordion").accordion("option", "active", 2);
}



//Trigger the signal to noise information panel
function focusSignalNoise()
{
  window.frames.info_frame.ShowSignalNoise();
}

//Clear any other focus related stuff
function focusClear()
{
  window.frames.info_frame.ClearFocus();
}

//Show the tables for the gratings of the given instrument
function focusGratings()
{
  focusClear();
  window.frames.info_frame.ShowGrating();
}


function onAccordionChange(e, ui) {
  //Called when the accordion navigation chnges.
  switch (ui.newHeader.attr('id')) {
    case "accordion_step1":
      window.frames.info_frame.ShowInstrument();
      break;
    case "accordion_step2":
      window.frames.info_frame.ShowOutputHelp();
      break;
    case "accordion_step3":
      if (outputType == "1") {
        $("#sn").focus();
      } else {
        $("#time").focus();
      }
      $("#floating_column").height($("#accordion").height());
      $("#info_frame").height($("#accordion").height() - 60);
      break;
    default:
      break;

  }
}


//Since the gratings are different between red and blue
//make a little script that fills them in as instruments are selected
function FillRedGratings() {
  var RedGratings = {
    "nograting": "Select Grating...",
    "150gpm": "150gpm",
    "270gpm": "270gpm",
    "300gpm": "300gpm",
    "600_4800": "600gpm/4800",
    "600_6310": "600gpm/6310",
    "1200_7700": "1200gpm/7700",
    "1200_9000": "1200gpm/9000"
  };
  var $el = $("#grating_selector");
  $el.empty();
  $.each(RedGratings, function(value, key) {
    $el.append($("<option></option>")
      .attr("value", value).text(key));
  });
  $el.prop('disabled', false);
}

function FillBlueGratings() {
  var BlueGratings = {
    "nograting": "Select Grating...",
    "300gpm": "300gpm",
    "500gpm": "500gpm",
    "800gpm": "800gpm",
    "600gpm": "600gpm",
    "832gpm": "832gpm",
    "1200gpm": "1200gpm"
  };
  var $el = $("#grating_selector");
  $el.empty();
  $.each(BlueGratings, function(value, key) {
    $el.append($("<option></option>")
      .attr("value", value).text(key));
  });
  $el.prop('disabled', false);

}

//And this is fired if the user selects no instrument
function ClearGratings() {

  var $el = $("#grating_selector");
  $el.empty();
  var option = $("<option></option>").attr("value", "noinstrument")
    .text("Select Instrument...");
  $el.append(option);
  $el.prop('disabled', true);
}


//Clear out all existing Errors
function clearErrors() {
  $("input").each(function() {
    $(this).removeClass("error");
  });
  $("select").each(function() {
    $(this).removeClass("error");
  });
}
