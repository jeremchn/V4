(function () {
    "use strict";

    var randomizeArray = function (arg) {
        var array = arg.slice();
        var currentIndex = array.length, temporaryValue, randomIndex;

        while (0 !== currentIndex) {

            randomIndex = Math.floor(Math.random() * currentIndex);
            currentIndex -= 1;

            temporaryValue = array[currentIndex];
            array[currentIndex] = array[randomIndex];
            array[randomIndex] = temporaryValue;
        }

        return array;
    }
    // data for the main cards sparklines
    var sparklineData = [47, 45, 54, 38, 56, 24, 65, 31, 37, 39, 62, 51, 35, 41, 35, 27, 93, 53, 61, 27, 54, 43, 19, 46];


    /* Total Customers */
    var options1 = {
        series: [{
            data: randomizeArray(sparklineData)
        }],
        labels: [...Array(24).keys()].map(n => `2018-09-0${n + 1}`),
        chart: {
            type: 'area',
            height: 50,
            sparkline: {
                enabled: true
            },
        },
        stroke: {
            curve: 'smooth',
            width: 1.5,
        },
        colors: ["var(--primary-color)"],
        fill: {
            type: ['gradient'],
            gradient: {
                shadeIntensity: 1,
                opacityFrom: 0.4,
                opacityTo: 0.1,
                stops: [0, 90, 100],
                colorStops: [
                    [
                        {
                            offset: 0,
                            color: "var(--primary01)",
                            opacity: 1
                        },
                        {
                            offset: 75,
                            color: "var(--primary005)",
                            opacity: 1
                        },
                        {
                            offset: 100,
                            color: 'var(--primary005)',
                            opacity: 0.05
                        }
                    ],
                ]
            }
        },
        tooltip: {
            fixed: {
                enabled: false
            },
            x: {
                show: false
            },
            y: {
                title: {
                    formatter: function (seriesName) {
                        return ''
                    }
                }
            },
            marker: {
                show: false
            }
        }
    };
    var chart1 = new ApexCharts(document.querySelector("#total-customers"), options1);
    chart1.render();
    /* Total Customers */

    /* Total Revenue */
    var options1 = {
        series: [{
            data: randomizeArray(sparklineData)
        }],
        labels: [...Array(24).keys()].map(n => `2018-09-0${n + 1}`),
        chart: {
            type: 'area',
            height: 50,
            sparkline: {
                enabled: true
            },
        },
        stroke: {
            curve: 'smooth',
            width: 1.5,
        },
        colors: ["rgb(255, 90, 41)"],
        fill: {
            type: ['gradient'],
            gradient: {
                shadeIntensity: 1,
                opacityFrom: 0.4,
                opacityTo: 0.1,
                stops: [0, 90, 100],
                colorStops: [
                    [
                        {
                            offset: 0,
                            color: "rgba(255, 90, 41, 0.1)",
                            opacity: 1
                        },
                        {
                            offset: 75,
                            color: "rgba(255, 90, 41, 0.05)",
                            opacity: 1
                        },
                        {
                            offset: 100,
                            color: '#ff5a29',
                            opacity: 0.05
                        }
                    ],
                ]
            }
        },
        tooltip: {
            fixed: {
                enabled: false
            },
            x: {
                show: false
            },
            y: {
                title: {
                    formatter: function (seriesName) {
                        return ''
                    }
                }
            },
            marker: {
                show: false
            }
        }
    };
    var chart1 = new ApexCharts(document.querySelector("#total-revenue"), options1);
    chart1.render();
    /* Total Revenue */

    /* Conversioon Ratio */
    var options1 = {
        series: [{
            data: randomizeArray(sparklineData)
        }],
        labels: [...Array(24).keys()].map(n => `2018-09-0${n + 1}`),
        chart: {
            type: 'area',
            height: 50,
            sparkline: {
                enabled: true
            },
        },
        stroke: {
            curve: 'smooth',
            width: 1.5,
        },
        colors: ["rgb(12, 199, 99)"],
        fill: {
            type: ['gradient'],
            gradient: {
                shadeIntensity: 1,
                opacityFrom: 0.4,
                opacityTo: 0.1,
                stops: [0, 90, 100],
                colorStops: [
                    [
                        {
                            offset: 0,
                            color: "rgba(12, 199, 99, 0.1)",
                            opacity: 1
                        },
                        {
                            offset: 75,
                            color: "rgba(12, 199, 99, 0.05)",
                            opacity: 1
                        },
                        {
                            offset: 100,
                            color: 'rgba(12, 199, 99, 0.05)',
                            opacity: 0.05
                        }
                    ],
                ]
            }
        },
        tooltip: {
            fixed: {
                enabled: false
            },
            x: {
                show: false
            },
            y: {
                title: {
                    formatter: function (seriesName) {
                        return ''
                    }
                }
            },
            marker: {
                show: false
            }
        }
    };
    var chart1 = new ApexCharts(document.querySelector("#conversion-ratio"), options1);
    chart1.render();
    /* Conversion Ratio */

    /* Total Deals */
    var options1 = {
        series: [{
            data: randomizeArray(sparklineData)
        }],
        labels: [...Array(24).keys()].map(n => `2018-09-0${n + 1}`),
        chart: {
            type: 'area',
            height: 50,
            sparkline: {
                enabled: true
            },
        },
        stroke: {
            curve: 'smooth',
            width: 1.5,
        },
        colors: ["rgb(12, 156, 252)"],
        fill: {
            type: ['gradient'],
            gradient: {
                shadeIntensity: 1,
                opacityFrom: 0.4,
                opacityTo: 0.1,
                stops: [0, 90, 100],
                colorStops: [
                    [
                        {
                            offset: 0,
                            color: "rgba(12, 156, 252, 0.1)",
                            opacity: 1
                        },
                        {
                            offset: 75,
                            color: "rgba(12, 156, 252, 0.05)",
                            opacity: 1
                        },
                        {
                            offset: 100,
                            color: 'rgba(12, 156, 252, 0.05)',
                            opacity: 0.05
                        }
                    ],
                ]
            }
        },
        tooltip: {
            fixed: {
                enabled: false
            },
            x: {
                show: false
            },
            y: {
                title: {
                    formatter: function (seriesName) {
                        return ''
                    }
                }
            },
            marker: {
                show: false
            }
        }
    };
    var chart1 = new ApexCharts(document.querySelector("#total-deals"), options1);
    chart1.render();
    /* Total Deals */

    /* industry_chart */
    document.addEventListener('DOMContentLoaded', () => {
        // Fonction pour charger les données CSV
        function fetchCSVData(filePath) {
            return fetch(filePath)
                .then(response => response.text())
                .then(csvText => {
                    return Papa.parse(csvText, {
                        header: true,
                        skipEmptyLines: true
                    }).data;
                });
        }
    
        // Fonction principale
        fetchCSVData('../../../companies_enriched.csv') // ← chemin vers ton CSV
            .then(data => {
                console.log("✅ Données CSV chargées :", data);
    
                const scoreCategories = [
                    "Location Score",
                    "Headcount Score",
                    "Industry Score",
                    "Company Type Score",
                    "Tech Score",
                    "Business Score"
                ];
    
                // Regrouper les données par industrie
                const industryGroups = {};
    
                data.forEach(company => {
                    const industry = company['Industry'] || 'Industrie inconnue';
                    if (!industryGroups[industry]) {
                        industryGroups[industry] = [];
                    }
                    industryGroups[industry].push(company);
                });
    
                // Calculer les moyennes des scores pour chaque industrie
                const series = Object.entries(industryGroups).map(([industry, companies]) => {
                    const averages = scoreCategories.map(score => {
                        const total = companies.reduce((sum, company) => {
                            const value = parseFloat(company[score]);
                            return sum + (isNaN(value) ? 0 : value);
                        }, 0);
                        return total / companies.length;
                    });
                    return {
                        name: industry,
                        data: averages
                    };
                });
    
                console.log("📊 Données pour le graphique radar :", series);
    
                // Configuration du graphique ApexCharts
                const options = {
                    series: series,
                    chart: {
                        height: 400,
                        type: 'radar'
                    },
                    xaxis: {
                        categories: scoreCategories
                    },
                    stroke: {
                        width: 2
                    },
                    fill: {
                        opacity: 0.2
                    },
                    tooltip: {
                        y: {
                            formatter: function (val) {
                                return val.toFixed(2);
                            }
                        }
                    },
                    legend: {
                        position: 'bottom'
                    }
                };
    
                const chartElement = document.querySelector("#features-industry-chart");
    
                if (chartElement) {
                    const chart = new ApexCharts(chartElement, options);
                    chart.render();
                } else {
                    console.error("L'élément #features-industry-chart est introuvable.");
                }
            })
            .catch(error => console.error("Erreur lors du chargement du CSV :", error));
    });
    /* industry_chart */
    


    

    /*  Project Analysis chart */
    var options = {
        series: [
            {
                name: "Total Income",
                data: [45, 30, 49, 45, 36, 42, 30, 35, 35, 54, 29, 36],
            },
            {
                name: "Total Expenses",
                data: [30, 35, 35, 30, 45, 25, 36, 54, 36, 29, 49, 42],
            },
            {
                name: "Total Deals",
                data: [45, 30, 49, 30, 45, 25, 36, 54, 36, 29, 49, 42],
            },
        ],
        chart: {
            type: "bar",
            height: 293,
            toolbar: {
                show: false,
            },
            dropShadow: {
                enabled: false,
            },
            stacked: true,
        },
        plotOptions: {
            bar: {
                columnWidth: "30%",
                borderRadiusApplication: "around",
                borderRadiusWhenStacked: "all",
                borderRadius: 3,
            },
        },
        responsive: [
            {
                breakpoint: 500,
                options: {
                    plotOptions: {
                        bar: {
                            columnWidth: "60%",
                        },
                    },
                },
            },
        ],
        stroke: {
            show: true,
            curve: "smooth",
            lineCap: "butt",
            width: [5, 5, 5],
            dashArray: 0,
        },
        grid: {
            borderColor: "#f5f4f4",
            strokeDashArray: 5,
            yaxis: {
                lines: {
                    show: true, 
                },
            },
        },
        colors: ["var(--primary-color)", "rgb(255, 90, 41)", "rgb(12, 199, 99)"],
        dataLabels: {
            enabled: false,
        },
        legend: {
            position: "top",
            markers: {
                size: 4,
                strokeWidth: 0,
                strokeColor: '#fff',
                fillColors: undefined,
                radius: 5,
                customHTML: undefined,
                onClick: undefined,
                offsetX: 0,
                offsetY: 0
              },
        },
        yaxis: {
            title: {
                style: {
                    color: "#adb5be",
                    fontSize: "14px",
                    fontFamily: "Montserrat, sans-serif",
                    fontWeight: 600,
                    cssClass: "apexcharts-yaxis-label",
                },
            },
            axisBorder: {
                show: true,
                color: "rgba(119, 119, 142, 0.05)",
                offsetX: 0,
                offsetY: 0,
            },
            axisTicks: {
                show: true,
                borderType: "solid",
                color: "rgba(119, 119, 142, 0.05)",
                width: 6,
                offsetX: 0,
                offsetY: 0,
            },
            labels: {
                formatter: function (y) {
                    return y.toFixed(0) + "";
                },
            },
        },
        xaxis: {
            type: "month",
            categories: [
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "sep",
                "oct",
                "nov",
                "dec",
            ],
            axisBorder: {
                show: false,
                color: "rgba(119, 119, 142, 0.05)",
                offsetX: 0,
                offsetY: 0,
            },
            axisTicks: {
                show: false,
                borderType: "solid",
                color: "rgba(119, 119, 142, 0.05)",
                width: 6,
                offsetX: 0,
                offsetY: 0,
            },
            labels: {
                rotate: -90,
            },
        },
    };
    var chart = new ApexCharts(document.querySelector("#project-analysis"), options);
    chart.render();
    /*  Project Analysis chart */

    // Fonction pour récupérer les données des régions et leurs pourcentages
function getRegionData() {
    const regionData = {};
    const regionElements = document.querySelectorAll('.crm-leads-channels-list .list-group-item');

    regionElements.forEach(item => {
        const regionName = item.querySelector('.flex-fill').textContent.trim(); // Récupère le nom de la région
        const percentageText = item.querySelector('.h6 span').textContent.trim(); // Récupère le pourcentage
        const percentage = parseFloat(percentageText.replace('%', '')); // Convertit en nombre
        regionData[regionName] = percentage;
    });

    return regionData;
}


document.addEventListener('DOMContentLoaded', () => {
    // Fonction pour récupérer les données des fichiers CSV
    function fetchCSVData(filePath) {
        return fetch(filePath)
            .then(response => response.text())
            .then(csvText => {
                return Papa.parse(csvText, {
                    header: true, // Utiliser la première ligne comme en-têtes
                    skipEmptyLines: true // Ignorer les lignes vides
                }).data;
            });
    }

    // Charger et analyser les données des fichiers CSV
    fetchCSVData('../../../companies_enriched.csv') // Chemin vers le fichier des entreprises
        .then(companiesData => {
            // Calculer le nombre d'entreprises par région
            const regionCounts = {};
            companiesData.forEach(company => {
                const region = company['Location'] || 'Région inconnue';
                if (!regionCounts[region]) {
                    regionCounts[region] = 0;
                }
                regionCounts[region]++;
            });

            // Trier les régions par nombre d'entreprises (descendant)
            const sortedRegions = Object.entries(regionCounts).sort((a, b) => b[1] - a[1]);

            // Calculer le total des entreprises
            const totalCompanies = companiesData.length;

            // Préparer les données pour le graphique
            const chartLabels = sortedRegions.map(([region]) => region);
            const chartData = sortedRegions.map(([_, count]) => count);

            // Vérifier les données
            console.log("Labels:", chartLabels);
            console.log("Data:", chartData);

            // Vérifier si les données sont valides
            if (chartLabels.length === 0 || chartData.length === 0) {
                console.error("Les données du graphique sont vides ou invalides.");
                return;
            }

            // Générer les lignes pour la liste des régions
            let regionRows = '';
            sortedRegions.forEach(([region, count]) => {
                const percentage = ((count / totalCompanies) * 100).toFixed(2);
                regionRows += `
                    <li class="list-group-item">
                        <div class="d-flex align-items-center gap-2">
                            <div class="flex-fill">${region}</div>
                            <div class="h6 mb-0 fw-semibold"><span class="me-2 fw-normal fs-13 d-inline-flex align-items-center">${percentage}%</span></div>
                        </div>
                    </li>
                `;
            });

            // Insérer les lignes dans la liste HTML
            const regionList = document.querySelector('.crm-leads-channels-list');
            if (regionList) {
                regionList.innerHTML = regionRows;
            } else {
                console.error("L'élément .crm-leads-channels-list est introuvable.");
            }

            // Créer le graphique avec ApexCharts
            const chartElement = document.querySelector("#leads-channels");
            if (chartElement) {
                const options = {
                    series: chartData, // Utiliser les données dynamiques
                    chart: {
                        height: 350,
                        type: "donut",
                    },
                    labels: chartLabels, // Utiliser les noms des régions dynamiques
                    plotOptions: {
                        pie: {
                            donut: {
                                size: "80%",
                                labels: {
                                    show: true,
                                    name: {
                                        show: true,
                                        fontSize: '16px',
                                        fontWeight: 600,
                                        color: '#495057',
                                    },
                                    value: {
                                        show: true,
                                        fontSize: '14px',
                                        fontWeight: 400,
                                        color: '#adb5bd',
                                    },
                                },
                            },
                        },
                    },
                    legend: {
                        position: 'bottom',
                    },
                    dataLabels: {
                        enabled: true, // Afficher les pourcentages sur le graphique
                        formatter: function (val) {
                            return val.toFixed(2) + "%";
                        },
                    },
                    tooltip: {
                        y: {
                            formatter: function (val) {
                                return val + " entreprises"; // Ajouter le nombre d'entreprises dans le tooltip
                            },
                        },
                    },
                    colors: [
                        "var(--primary-color)",
                        "rgb(255, 90, 41)",
                        "rgb(12, 199, 99)",
                        "rgb(12, 156, 252)",
                        "#775DD0",
                        "#546E7A",
                        "#26A69A",
                        "#D10CE8",
                    ],
                };

                const chart = new ApexCharts(chartElement, options);
                chart.render();
            } else {
                console.error("L'élément #leads-channels est introuvable.");
            }
        })
        .catch(error => console.error('Erreur lors du chargement des données :', error));
});
})();
