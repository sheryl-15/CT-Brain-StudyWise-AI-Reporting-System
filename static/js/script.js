const loadStudyBtn = document.getElementById("loadStudyBtn");
const analyzeBtn = document.getElementById("analyzeBtn");
const studyUidInput = document.getElementById("studyUidInput");
const seriesSelect = document.getElementById("seriesSelect");

const analysisLoader = document.getElementById("analysisLoader");
const progressFill = document.getElementById("progressFill");
const loaderStep = document.getElementById("loaderStep");

function showToast(message, type = "success") {
    const toast = document.getElementById("toast");
    toast.innerText = message;
    toast.className = "";
    toast.classList.add(type);
    toast.classList.add("show");

    setTimeout(() => {
        toast.classList.remove("show");
    }, 3000);
}

function showLoader(stepText, progress) {
    analysisLoader.classList.add("show");
    loaderStep.innerText = stepText;
    progressFill.style.width = progress + "%";
}

function updateLoader(stepText, progress) {
    loaderStep.innerText = stepText;
    progressFill.style.width = progress + "%";
}

function hideLoader() {
    setTimeout(() => {
        analysisLoader.classList.remove("show");
        progressFill.style.width = "0%";
        loaderStep.innerText = "Preparing analysis...";
    }, 500);
}

loadStudyBtn.addEventListener("click", async function () {
    const studyUid = studyUidInput.value.trim();

    if (!studyUid) {
        showToast("Please enter Study UID", "error");
        return;
    }

    loadStudyBtn.innerText = "Loading...";
    loadStudyBtn.disabled = true;

    try {
        const response = await fetch("/load-study", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                study_uid: studyUid
            })
        });

        const data = await response.json();

        if (!data.success) {
            showToast(data.error, "error");
            return;
        }

        document.getElementById("patientName").innerText = data.patient.name;
        document.getElementById("patientId").innerText = data.patient.id;
        document.getElementById("patientAge").innerText = data.patient.age;
        document.getElementById("patientGender").innerText = data.patient.gender;
        document.getElementById("studyDate").innerText = data.patient.study_date;

        document.getElementById("studyUid").innerText = data.study.study_uid;
        document.getElementById("studyDescription").innerText = data.study.description;
        document.getElementById("institution").innerText = data.study.institution;
        document.getElementById("doctorName").innerText = data.study.doctor;
        document.getElementById("totalSeries").innerText = data.study.total_series;

        document.getElementById("reportStudyUid").innerText = data.study.study_uid;

        seriesSelect.innerHTML = "";

        data.series.forEach((series, index) => {
            const option = document.createElement("option");

            option.value = series.series_uid;
            option.textContent =
                `${index + 1}. ${series.description} | ${series.modality} | ${series.instances} slices`;

            option.dataset.description = series.description;
            option.dataset.instances = series.instances;

            seriesSelect.appendChild(option);
        });

        showToast("Study loaded successfully");

    } catch (error) {
        showToast("Error loading study", "error");
    } finally {
        loadStudyBtn.innerText = "Load Study";
        loadStudyBtn.disabled = false;
    }
});


analyzeBtn.addEventListener("click", async function () {
    const studyUid = studyUidInput.value.trim();
    const selectedOption = seriesSelect.options[seriesSelect.selectedIndex];

    if (!studyUid) {
        showToast("Please load a study first", "error");
        return;
    }

    if (!selectedOption || !selectedOption.value) {
        showToast("Please select a series", "error");
        return;
    }

    const seriesUid = selectedOption.value;
    const seriesDescription = selectedOption.dataset.description || "--";

    analyzeBtn.innerText = "Analyzing...";
    analyzeBtn.disabled = true;

    document.getElementById("predictionText").innerText = "Processing...";
    document.getElementById("predictionSubtext").innerText =
        "AI analysis is currently running.";

    showLoader("Downloading DICOM series from PACS...", 20);

    try {
        setTimeout(() => updateLoader("Extracting CT image features...", 55), 1200);
        setTimeout(() => updateLoader("Running AI prediction model...", 80), 2500);

        const response = await fetch("/analyze-series", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                study_uid: studyUid,
                series_uid: seriesUid,
                series_description: seriesDescription
            })
        });

        const data = await response.json();

        if (!data.success) {
            hideLoader();
            showToast(data.error, "error");
            return;
        }

        updateLoader("Prediction completed successfully.", 100);

        if (data.metadata) {
            document.getElementById("patientName").innerText = data.metadata.patient_name;
            document.getElementById("patientId").innerText = data.metadata.patient_id;
            document.getElementById("patientAge").innerText = data.metadata.patient_age;
            document.getElementById("patientGender").innerText = data.metadata.patient_gender;
            document.getElementById("studyDate").innerText = data.metadata.study_date;

            document.getElementById("studyDescription").innerText = data.metadata.study_description;
            document.getElementById("institution").innerText = data.metadata.institution;
            document.getElementById("doctorName").innerText = data.metadata.doctor;
        }

        const predictionText = document.getElementById("predictionText");
        const predictionBox = document.querySelector(".prediction-box");

        predictionText.innerText = data.prediction;
        predictionBox.classList.remove("normal-result", "abnormal-result");

        if (data.prediction === "ABNORMAL") {
            predictionBox.classList.add("abnormal-result");
        } else {
            predictionBox.classList.add("normal-result");
        }

        document.getElementById("predictionSubtext").innerText =
            "AI prediction completed for selected CT series.";

        document.getElementById("probability").innerText = data.probability + "%";
        document.getElementById("riskLevel").innerText = data.risk;
        document.getElementById("sliceCount").innerText = data.slice_count;
        document.getElementById("selectedSeries").innerText = data.series_description;

        document.getElementById("reportSeriesUid").innerText = data.series_uid;
        document.getElementById("reportPrediction").innerText = data.prediction;
        document.getElementById("generatedTime").innerText = data.generated_time;

        const previewContainer = document.querySelector(".slice-preview");
        previewContainer.innerHTML = "";

        if (data.preview_urls && data.preview_urls.length > 0) {
            data.preview_urls.forEach((url, index) => {
                const div = document.createElement("div");
                div.innerHTML = `<img src="${url}" alt="CT Slice ${index + 1}">`;
                previewContainer.appendChild(div);
            });
        } else {
            previewContainer.innerHTML = `
                <div><p>No Preview Available</p></div>
                <div><p>No Preview Available</p></div>
                <div><p>No Preview Available</p></div>
            `;
        }

        hideLoader();
        showToast("Prediction completed successfully");

    } catch (error) {
        hideLoader();
        showToast("Error during analysis", "error");
    } finally {
        analyzeBtn.innerText = "Analyze Selected Series";
        analyzeBtn.disabled = false;
    }
});


const downloadReportBtn = document.getElementById("downloadReportBtn");

downloadReportBtn.addEventListener("click", function () {
    const studyUid = document.getElementById("reportStudyUid").innerText;
    const seriesUid = document.getElementById("reportSeriesUid").innerText;
    const prediction = document.getElementById("reportPrediction").innerText;
    const generatedTime = document.getElementById("generatedTime").innerText;

    const probability = document.getElementById("probability").innerText;
    const riskLevel = document.getElementById("riskLevel").innerText;
    const sliceCount = document.getElementById("sliceCount").innerText;
    const selectedSeries = document.getElementById("selectedSeries").innerText;

    if (prediction === "--") {
        showToast("Please analyze a series before downloading report", "error");
        return;
    }

    const reportContent = `
CT BRAIN STUDY-WISE AI REPORT
========================================

Study UID:
${studyUid}

Series UID:
${seriesUid}

Selected Series:
${selectedSeries}

Prediction:
${prediction}

Abnormal Probability:
${probability}

Risk Level:
${riskLevel}

Slices Analysed:
${sliceCount}

Generated Time:
${generatedTime}

----------------------------------------
This AI-generated result is intended for clinical decision support only
and must be verified by a qualified radiologist.
`;

    const blob = new Blob([reportContent], { type: "text/plain" });
    const url = URL.createObjectURL(blob);

    const link = document.createElement("a");
    link.href = url;
    link.download = "CT_Brain_AI_Report.txt";
    link.click();

    URL.revokeObjectURL(url);

    showToast("Report downloaded successfully");
});