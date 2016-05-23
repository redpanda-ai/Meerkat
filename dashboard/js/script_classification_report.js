d3.csv(csvFileName, function(error, data) {
  if (error) throw error;

  var baseWidth = data.length < 15 ? 1000 : data.length * 60 + 100;
  var margin = {top: 30, right: 20, bottom: 30, left: 80},
    width = baseWidth - margin.left - margin.right,
    height = 400 - margin.top - margin.bottom;

  var x0 = d3.scale.ordinal();

  var x1 = d3.scale.ordinal();

  var y = d3.scale.linear()
    .range([height, 0]);

  var color = d3.scale.ordinal()
    .range(["#98abc5", "#7b6888", "#d0743c", "#ff8c00"]);

  var xAxis = d3.svg.axis()
    .scale(x0)
    .orient("bottom");

  var formatPercent = d3.format(".0%");
  var yAxis = d3.svg.axis()
    .scale(y)
    .orient("left")
    .tickFormat(formatPercent);

  var svg = d3.select("#svg-chart").append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom + 300)
    .append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

  var statNames = ['Precision', 'Recall', 'Specificity', 'F Measure'];

  data.forEach(function(d) {
    d.stats = statNames.map(function(name) { return {name: name, value: +d[name]}; });
  });

  minVal = d3.min(data, function(d) {
    return Math.min(d.stats[0].value, d.stats[1].value, d.stats[2].value, d.stats[3].value); 
  });

  x0.rangeRoundBands([0, data.length * 60], 0);
  x0.domain(data.map(function(d) { return d.Class; }));
  x1.domain(statNames).rangeRoundBands([10, 50]);
  y.domain([minVal - 0.05, d3.max(data, function(d) { return d3.max(d.stats, function(d) { return d.value; }); })]);

  svg.append("g")
    .attr("class", "x axis")
    .attr("transform", "translate(0," + (height + 100) + ")")
    .call(xAxis)
    .selectAll("text")
    .attr("y", 0)
    .attr("x", 9)
    .attr("dy", ".35em")
    .attr("transform", "rotate(90)")
    .style("text-anchor", "start");

  svg.append("g")
    .attr("class", "y axis")
    .attr("transform", "translate(0, 100)")
    .call(yAxis);

  var className = svg.selectAll(".className")
    .data(data)
    .enter().append("g")
    .attr("class", "g")
    .attr("y", 100)
    .attr("transform", function(d) { return "translate(" + x0(d.Class) + ",0)"; });

  className.selectAll("rect")
    .data(function(d) { return d.stats; })
    .enter().append("rect")
    .attr("width", x1.rangeBand() - 5)
    .attr("x", function(d) { return x1(d.name); })
    .attr("y", function(d) { return y(d.value) + 100; })
    .attr("height", function(d) { return height - y(d.value); })
    .style("fill", function(d) { return color(d.name); });

  d3.selectAll("input").on("change", change);

  var legend = svg.selectAll(".legend")
    .data(statNames.slice())
    .enter().append("g")
    .attr("class", "legend")
    .attr("transform", function(d, i) { return "translate(0," + i * 20 + ")"; });

  legend.append("rect")
    .attr("x",  560)
    .attr("width", 18)
    .attr("height", 18)
    .style("fill", color);

  legend.append("text")
    .attr("x", 550)
    .attr("y", 9)
    .attr("dy", ".35em")
    .style("text-anchor", "end")
    .text(function(d) { return d; });

  var names = csvFileName.split("_");
  var titleName = "Classification Report - " + names[3] + " " + names[4] + " " + names[5] + " ";
  if(names[3] == "subtype") {
    titleName += names[6].split(".")[0];
  }

  svg.append("text")           
    .attr("y", 40)
    .style("font-size", "15px") 
    .attr("class", "title") 
    .text(titleName);

  function change() {
    var sortKey;
    switch(this.value) {
      case "Precision": sortKey = function(a, b) { return b.stats[0].value - a.stats[0].value; }; break;
      case "Recall": sortKey = function(a, b) { return b.stats[1].value - a.stats[1].value; }; break;
      case "Specificity": sortKey = function(a, b) { return b.stats[2].value - a.stats[2].value; }; break;
      case "FMeasure": sortKey = function(a, b) { return b.stats[3].value - a.stats[3].value; }; break;
      default: sortKey = function(a, b) { return d3.ascending(a.Class, b.Class); }
    }

    var x2 = x0.domain(data.sort(sortKey)
      .map(function(d) { return d.Class; }))
      .copy();

    var transition = svg.transition().duration(500),
      delay = function(d, i) { return i * 1; };

    transition.select(".x.axis")
      .call(xAxis)
      .selectAll("g")
      .delay(delay)
      .selectAll("text")
      .attr("y", 0)
      .attr("x", 9)
      .attr("dy", ".35em")
      .attr("transform", "rotate(90)")
      .style("text-anchor", "start");

    transition.selectAll("g.g")
      .delay(delay)
      .attr("transform", function(d) { return "translate(" + x2(d.Class) + ",0)"; });
  };

  function pieGraph(id, tF, type){
    function segColor(c){ return {
      0.9: "#effdde",
      0.95: "#dafac8",
      0.97: "#90ee90",
      0.99: "#539288",
      1: "#406685" }[c]; 
    };

    function pie(pD) {
      var pC ={},    
        pieDim ={w:200, h: 200};
      pieDim.r = Math.min(pieDim.w, pieDim.h) / 2;

      var piesvg = d3.select(id).append("svg")
        .attr("width", pieDim.w).
        attr("height", pieDim.h).
        append("g")
        .attr("transform", "translate("+pieDim.w/2+","+pieDim.h/2+")");
        
      var arc = d3.svg.arc().outerRadius(pieDim.r - 10).innerRadius(0);

      var pie = d3.layout.pie().sort(null).value(function(d) { return d.freq; });

      piesvg.selectAll("path").
        data(pie(pD)).
        enter().append("path").
        attr("d", arc)
        .each(function(d) { this._current = d; })
        .style("fill", function(d) { return segColor(d.data.type); });
    };

    function legend(lD) {
      var leg = {};
            
      var legend = d3.select(id).
        append("table").
        attr('class','legend');
          
      var title = legend.append("thead").
        append("tr").
        append("th").
        text(type).
        attr("y", 20);

      var tr = legend.append("tbody").
        attr("class", "legendBody").
        selectAll("tr").
        data(lD).
        enter().append("tr");

      tr.append("td").
        append("svg").
        attr("width", '16').
        attr("height", '16').
        append("rect").
        attr("width", '16').
        attr("height", '16').
        attr("fill",function(d){ return segColor(d.type); });
            
      tr.append("td")
        .attr("width", 80)
        .text(function(d) { 
          switch(d.type) {
            case "0.9": return "0 - 89.9%";
            case "0.95": return "90 - 94.9%";
            case "0.97": return "95 - 96.9%";
            case "0.99": return "97 - 98.9%";
            default: return "99 - 100%";
          }
        });

      tr.append("td").
        attr("class",'legendFreq')
        .text(function(d){ return d3.format(",")(d.freq);});

      tr.append("td").
        attr("class",'legendPerc')
        .text(function(d){ return getLegend(d,lD);});

      function getLegend(d,aD){ 
        return d3.format("%")(d.freq/d3.sum(aD.map(function(v){ return v.freq; })));
      };
    };

    pie(tF);
    legend(tF);  
  };

  var statPie = [];
  for(var j = 0; j < 4; j++) {
    statPie[j] = [{"type": "0.9", "freq": 0}, {"type": "0.95", "freq": 0}, {"type": "0.97", "freq": 0},
      {"type": "0.99", "freq": 0}, {"type": "1", "freq": 0}];
  }
  for(var i = 0; i < data.length; i++) {
    for(var j = 0; j < 4; j++) {
      var statVal = data[i].stats[j].value;
      if(statVal < 0.90) {
        statPie[j][0]["freq"] += 1;
      } else if(statVal >= 0.90 && statVal < 0.95) {
        statPie[j][1]["freq"] += 1;
      } else if(statVal >= 0.95 && statVal < 0.97) {
        statPie[j][2]["freq"] += 1;
      } else if(statVal >= 0.97 && statVal < 0.99) {
        statPie[j][3]["freq"] += 1;
      } else {
        statPie[j][4]["freq"] += 1;
      }
    }
  }

  pieGraph('#pie-chart-left-col-1', statPie[0], "Precision");
  pieGraph('#pie-chart-right-col-1', statPie[1], "Recall");
  pieGraph('#pie-chart-left-col-2', statPie[2], "Specificity");
  pieGraph('#pie-chart-right-col-2', statPie[3], "F Measure");
});