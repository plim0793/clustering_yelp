      function myFunction () {
        var pref = $('#preference').val();
      $.ajax({
       type: "POST",
       contentType: "application/json; charset=utf-8",
       url: "/",
       dataType: "json",
       async: true,
       data: '{"preference": "'+pref+'"}',
       success: function (data) {
        console.log(data);

        $('#preference').text(data.preference);

        },
        error: function (result) {
       }
      })
      }; 