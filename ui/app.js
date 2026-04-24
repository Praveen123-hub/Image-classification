Dropzone.autoDiscover = false;

function init() {
    let dz = new Dropzone("#dropzone", {
        url: "/classify_image",
        maxFiles: 1,
        addRemoveLinks: true,
        dictDefaultMessage: "Drop image here to classify",
        autoProcessQueue: false
    });
    
    dz.on("addedfile", function() {
        if (dz.files[1] != null) {
            dz.removeFile(dz.files[0]);        
        }
    });

    $("#submitBtn").on('click', function (e) {
        if (dz.files.length > 0) {
            const file = dz.files[0];
            const reader = new FileReader();
            
            reader.onload = function(event) {
                const imageData = event.target.result;
                const url = "http://127.0.0.1:5000/classify_image";

                $.post(url, { image_data: imageData }, function(data, status) {
                    console.log(data);
                    if (!data || data.length == 0) {
                        $("#resultHolder").hide();
                        $("#divClassTable").hide();                
                        $("#error").show();
                        return;
                    }

                    let match = data[0]; // Take the first detected face
                    if (match) {
                        $("#error").hide();
                        $("#resultHolder").show();
                        $("#divClassTable").show();
                        $("#resultHolder").html($(`[data-player="${match.class}"]`).html());
                        
                        let classDictionary = match.class_dictionary;
                        for(let personName in classDictionary) {
                            let index = classDictionary[personName];
                            let probabilityScore = match.class_probability[index];
                            $("#score_" + personName).html(probabilityScore + "%");
                        }
                    }
                });
            };
            reader.readAsDataURL(file);
        }
    });
}

$(document).ready(function() {
    $("#error").hide();
    $("#resultHolder").hide();
    $("#divClassTable").hide();
    init();
});