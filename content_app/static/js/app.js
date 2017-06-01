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
        $('#top1').text(data.top_1);
        $('#top2').text(data.top_2);
        $('#top3').text(data.top_3);
        $('#top4').text(data.top_4);
        $('#top5').text(data.top_5);
        $('#top1r').text(data.top_1_rev);
        $('#top2r').text(data.top_2_rev);
        $('#top3r').text(data.top_3_rev);
        $('#top4r').text(data.top_4_rev);
        $('#top5r').text(data.top_5_rev);

        },
        error: function (result) {
       }
      })
      }; 