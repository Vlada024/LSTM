// ============================================================================
// LSTM DATASET EXPLORER - Main Application
// ============================================================================

// ============================================================================
// STATE
// ============================================================================
let dataset = [];
let currentViewerStart = 0;
let previewChartInstance = null;
let viewerChartInstance = null;
let lossChartInstance = null;
let valLossChartInstance = null;
let stopTraining = false;

// ============================================================================
// DOM ELEMENTS
// ============================================================================
const samplesInput = document.getElementById('samples');
const seqLenInput = document.getElementById('seqLen');
const noiseStdInput = document.getElementById('noiseStd');
const seedInput = document.getElementById('seed');
const ampMinInput = document.getElementById('ampMin');
const ampMaxInput = document.getElementById('ampMax');
const freqMinInput = document.getElementById('freqMin');
const freqMaxInput = document.getElementById('freqMax');
const generateBtn = document.getElementById('generateBtn');
const downloadBtn = document.getElementById('downloadBtn');
const summaryDiv = document.getElementById('summary');

const previewChart = document.getElementById('previewChart');
const viewerSamplesInput = document.getElementById('viewerSamples');
const viewerStartInput = document.getElementById('viewerStart');
const overlayChk = document.getElementById('overlayChk');
const renderBtn = document.getElementById('renderBtn');
const prevBtn = document.getElementById('prevBtn');
const nextBtn = document.getElementById('nextBtn');
const randomBtn = document.getElementById('randomBtn');

const lstmUnitsInput = document.getElementById('lstmUnits');
const batchSizeInput = document.getElementById('batchSize');
const epochsInput = document.getElementById('epochs');
const learningRateInput = document.getElementById('learningRate');
const valSplitInput = document.getElementById('valSplit');
const startTrainBtn = document.getElementById('startTrainBtn');
const stopTrainBtn = document.getElementById('stopTrainBtn');
const trainStatus = document.getElementById('trainStatus');
const trainChartsContainer = document.getElementById('trainChartsContainer');
const lossChart = document.getElementById('lossChart');
const valLossChart = document.getElementById('valLossChart');

const showArchBtn = document.getElementById('showArchBtn');
const archModal = document.getElementById('archModal');
const archText = document.getElementById('archText');

// ============================================================================
// DATASET GENERATION
// ============================================================================

/**
 * Generate synthetic sine-wave dataset (async with chunking)
 */
async function generateSineDataset(params, progressCallback) {
  const { samples, seqLen, noiseStd, seed, ampMin, ampMax, freqMin, freqMax } = params;
  const data = [];
  const chunkSize = 500;
  
  for (let start = 0; start < samples; start += chunkSize) {
    const end = Math.min(start + chunkSize, samples);
    
    // Process chunk
    for (let i = start; i < end; i++) {
      const amp = ampMin + Math.random() * (ampMax - ampMin);
      const freq = freqMin + Math.random() * (freqMax - freqMin);
      const phase = Math.random() * 2 * Math.PI;

      const sequence = [];
      for (let t = 0; t < seqLen; t++) {
        let value = amp * Math.sin(2 * Math.PI * freq * t + phase);
        value += noiseStd * (Math.random() * 2 - 1);
        sequence.push(value);
      }
      data.push(sequence);
    }
    
    // Report progress
    if (progressCallback) {
      const progress = Math.round((end / samples) * 100);
      progressCallback(progress);
    }
    
    // Yield to event loop after each chunk
    await new Promise(resolve => setTimeout(resolve, 0));
  }

  return data;
}

/**
 * Validate dataset generation parameters
 */
function validateDatasetParams(params) {
  if (params.samples < 1 || params.samples > 20000) {
    throw new Error('Samples must be between 1 and 20,000');
  }
  if (params.seqLen < 5) {
    throw new Error('Sequence length must be at least 5');
  }
  if (params.noiseStd < 0) {
    throw new Error('Noise std must be non-negative');
  }
  if (params.ampMin >= params.ampMax) {
    throw new Error('Amplitude min must be less than max');
  }
  if (params.freqMin >= params.freqMax) {
    throw new Error('Frequency min must be less than max');
  }
}

/**
 * Handle Generate Dataset button click
 */
generateBtn.onclick = async () => {
  try {
    const params = {
      samples: parseInt(samplesInput.value),
      seqLen: parseInt(seqLenInput.value),
      noiseStd: parseFloat(noiseStdInput.value),
      seed: parseInt(seedInput.value),
      ampMin: parseFloat(ampMinInput.value),
      ampMax: parseFloat(ampMaxInput.value),
      freqMin: parseFloat(freqMinInput.value),
      freqMax: parseFloat(freqMaxInput.value)
    };

    validateDatasetParams(params);

    // Disable button during generation
    generateBtn.disabled = true;
    summaryDiv.innerHTML = '<strong>Generating dataset...</strong> 0%';

    // Generate dataset with progress updates
    dataset = await generateSineDataset(params, (progress) => {
      summaryDiv.innerHTML = `<strong>Generating dataset...</strong> ${progress}%`;
    });

    // Update summary
    summaryDiv.innerHTML = `
      <strong>Dataset Generated Successfully</strong><br>
      Samples: ${params.samples} | Seq Length: ${params.seqLen} | 
      Noise Std: ${params.noiseStd} | Seed: ${params.seed}<br>
      Amplitude: [${params.ampMin}, ${params.ampMax}] | 
      Frequency: [${params.freqMin}, ${params.freqMax}]
    `;

    // Enable download button
    downloadBtn.disabled = false;

    // Reset viewer and render preview
    currentViewerStart = 0;
    viewerStartInput.value = 0;
    viewerSamplesInput.value = 5;
    renderPreview();

  } catch (error) {
    showStatus(`Error: ${error.message}`, 'error');
  } finally {
    generateBtn.disabled = false;
  }
};

/**
 * Handle Download JSON button click
 */
downloadBtn.onclick = () => {
  if (dataset.length === 0) return;

  const dataStr = JSON.stringify(dataset, null, 2);
  const blob = new Blob([dataStr], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `sine_dataset_${new Date().toISOString().slice(0, 10)}.json`;
  a.click();
  URL.revokeObjectURL(url);

  showStatus('Dataset downloaded successfully', 'success');
};

// ============================================================================
// PREVIEW GRAPH (Below Generate Button)
// ============================================================================

/**
 * Render preview graph showing configurable sample sequences
 */
function renderPreview() {
  if (dataset.length === 0) return;

  const numSamples = Math.min(parseInt(viewerSamplesInput.value) || 5, dataset.length);
  currentViewerStart = Math.min(currentViewerStart, dataset.length - numSamples);
  viewerStartInput.value = currentViewerStart;

  const labels = Array.from({ length: dataset[0].length }, (_, i) => i);
  const colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#00f2fe', '#fa709a', '#fee140', '#30cfd0', '#a8edea', '#fed6e3'];
  const datasets = [];
  const isOverlay = overlayChk.checked;

  for (let i = 0; i < numSamples; i++) {
    const idx = currentViewerStart + i;
    if (idx < dataset.length) {
      datasets.push({
        label: `Sample ${idx}`,
        data: dataset[idx],
        borderColor: isOverlay ? 'rgba(102, 126, 234, 0.6)' : colors[i % colors.length],
        borderWidth: isOverlay ? 2 : 2.5,
        fill: false,
        pointRadius: 0,
        tension: 0.3
      });
    }
  }

  if (previewChartInstance) {
    previewChartInstance.destroy();
  }

  previewChartInstance = new Chart(previewChart.getContext('2d'), {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: false,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: true, position: 'top' },
        title: { display: false }
      },
      scales: {
        x: {
          title: { display: true, text: 'Time Step' },
          grid: { display: false }
        },
        y: {
          title: { display: true, text: 'Amplitude' }
        }
      }
    }
  });
}

// Viewer controls
renderBtn.onclick = renderPreview;
viewerSamplesInput.onchange = renderPreview;
overlayChk.onchange = renderPreview;

prevBtn.onclick = () => {
  const numSamples = parseInt(viewerSamplesInput.value) || 5;
  currentViewerStart = Math.max(0, currentViewerStart - 1);
  renderPreview();
};

nextBtn.onclick = () => {
  const numSamples = parseInt(viewerSamplesInput.value) || 5;
  currentViewerStart = Math.min(dataset.length - numSamples, currentViewerStart + 1);
  renderPreview();
};

randomBtn.onclick = () => {
  const numSamples = parseInt(viewerSamplesInput.value) || 5;
  currentViewerStart = Math.floor(Math.random() * Math.max(1, dataset.length - numSamples + 1));
  renderPreview();
};

// ============================================================================
// DATASET VIEWER GRAPH
// ============================================================================

/**
 * Render dataset viewer graph
 */
function renderViewer() {
  if (dataset.length === 0) return;

  const numSamples = parseInt(viewerSamplesInput.value) || 1;
  currentViewerStart = Math.min(currentViewerStart, dataset.length - numSamples);
  viewerStartInput.value = currentViewerStart;

  const labels = Array.from({ length: dataset[0].length }, (_, i) => i);
  const datasets = [];
  const isOverlay = overlayChk.checked;

  for (let i = 0; i < numSamples; i++) {
    const idx = currentViewerStart + i;
    if (idx < dataset.length) {
      datasets.push({
        label: `Sample ${idx}`,
        data: dataset[idx],
        borderColor: isOverlay ? '#667eea' : `hsl(${(idx * 50) % 360}, 70%, 60%)`,
        borderWidth: isOverlay ? 2.5 : 2,
        fill: false,
        pointRadius: 0,
        tension: 0.3,
        opacity: isOverlay ? 0.8 : 1
      });
    }
  }

  if (viewerChartInstance) {
    viewerChartInstance.destroy();
  }

  viewerChartInstance = new Chart(viewerChart.getContext('2d'), {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: true, position: 'top' },
        title: { display: false }
      },
      scales: {
        x: {
          title: { display: true, text: 'Time Step' },
          grid: { display: false }
        },
        y: {
          title: { display: true, text: 'Amplitude' }
        }
      }
    }
  });
}

// ============================================================================
// LSTM TRAINING
// ============================================================================

/**
 * Show status message
 */
function showStatus(message, type = 'info') {
  trainStatus.textContent = message;
  trainStatus.className = `status show ${type}`;
  // Keep epoch progress visible; only auto-hide non-epoch messages
  if (!message.includes('Epoch')) {
    setTimeout(() => {
      trainStatus.classList.remove('show');
    }, 5000);
  }
}

/**
 * Handle Start Training button
 */
startTrainBtn.onclick = async () => {
  if (dataset.length === 0) {
    showStatus('Please generate a dataset first', 'error');
    return;
  }

  try {
    showStatus('Preparing training data...', 'info');
    startTrainBtn.disabled = true;
    stopTrainBtn.disabled = false;
    stopTraining = false;

    const units = parseInt(lstmUnitsInput.value);
    const batchSize = parseInt(batchSizeInput.value);
    const epochs = parseInt(epochsInput.value);
    const lr = parseFloat(learningRateInput.value);
    const valSplit = parseFloat(valSplitInput.value);

    // Prepare training data efficiently with Float32Arrays
    const numSamples = dataset.length;
    const seqLength = dataset[0].length - 1; // X uses all but last value
    
    // Find min and max for normalization
    let minVal = Infinity;
    let maxVal = -Infinity;
    for (let i = 0; i < numSamples; i++) {
      for (let t = 0; t < dataset[i].length; t++) {
        const val = dataset[i][t];
        if (val < minVal) minVal = val;
        if (val > maxVal) maxVal = val;
      }
    }
    const range = maxVal - minVal;
    
    // Pre-allocate typed arrays for better performance
    const xData = new Float32Array(numSamples * seqLength * 1);
    const yData = new Float32Array(numSamples * 1);
    
    // Fill arrays with normalization and async yielding for large datasets
    for (let i = 0; i < numSamples; i++) {
      const seq = dataset[i];
      
      // Fill X data (all values except last) - normalized to [0, 1]
      for (let t = 0; t < seqLength; t++) {
        xData[i * seqLength + t] = (seq[t] - minVal) / range;
      }
      
      // Fill Y data (last value) - normalized to [0, 1]
      yData[i] = (seq[seq.length - 1] - minVal) / range;
      
      // Yield to main thread every 1000 iterations
      if (i > 0 && i % 1000 === 0) {
        showStatus(`Preparing training data... ${Math.round((i / numSamples) * 100)}%`, 'info');
        await new Promise(resolve => setTimeout(resolve, 0));
      }
    }
    
    showStatus('Creating tensors...', 'info');
    await new Promise(resolve => setTimeout(resolve, 10));

    // Create tensors with proper shapes
    const xsTensor = tf.tensor3d(xData, [numSamples, seqLength, 1]);
    const ysTensor = tf.tensor2d(yData, [numSamples, 1]);

    // Build LSTM model
    const model = tf.sequential({
      layers: [
        tf.layers.lstm({ units, inputShape: [seqLength, 1], activation: 'relu' }),
        tf.layers.dropout({ rate: 0.2 }),
        tf.layers.dense({ units: 32, activation: 'relu' }),
        tf.layers.dense({ units: 1, activation: 'linear' })
      ]
    });

    model.compile({
      optimizer: tf.train.adam(lr),
      loss: 'meanSquaredError',
      metrics: ['mae']
    });

    // Initialize training charts
    trainChartsContainer.style.display = 'grid';

    if (lossChartInstance) lossChartInstance.destroy();
    if (valLossChartInstance) valLossChartInstance.destroy();

    const trainLosses = [];
    const valLosses = [];

    lossChartInstance = new Chart(lossChart.getContext('2d'), {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          {
            label: 'Training Loss',
            data: trainLosses,
            borderColor: '#667eea',
            borderWidth: 2,
            fill: false,
            pointRadius: 0,
            tension: 0.3
          }
        ]
      },
      options: {
        responsive: false,
        maintainAspectRatio: false,
        animation: false,
        plugins: { legend: { display: true } },
        scales: { 
          y: { 
            type: 'logarithmic',
            title: { display: true, text: 'Loss (log scale)' }
          } 
        }
      }
    });

    valLossChartInstance = new Chart(valLossChart.getContext('2d'), {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          {
            label: 'Validation Loss',
            data: valLosses,
            borderColor: '#e74c3c',
            borderWidth: 2,
            fill: false,
            pointRadius: 0,
            tension: 0.3
          }
        ]
      },
      options: {
        responsive: false,
        maintainAspectRatio: false,
        animation: false,
        plugins: { legend: { display: true } },
        scales: { 
          y: { 
            type: 'logarithmic',
            title: { display: true, text: 'Loss (log scale)' }
          } 
        }
      }
    });

    // Train model
    showStatus('Training in progress...', 'info');

    // Throttle variables for chart updates
    let lastChartUpdate = 0;
    const chartUpdateThrottle = 200; // ms

    // Use yieldEvery parameter for automatic UI yielding
    await model.fit(xsTensor, ysTensor, {
      batchSize,
      epochs,
      validationSplit: valSplit,
      verbose: 0,
      shuffle: true,
      yieldEvery: 'auto',
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          trainLosses.push(logs.loss);
          valLosses.push(logs.val_loss);

          const now = Date.now();
          const isFinalEpoch = epoch === epochs - 1;
          const shouldUpdateCharts = (now - lastChartUpdate >= chartUpdateThrottle) || isFinalEpoch;

          if (shouldUpdateCharts) {
            lossChartInstance.data.labels.push(epoch + 1);
            lossChartInstance.data.datasets[0].data = [...trainLosses];
            lossChartInstance.update('none');

            valLossChartInstance.data.labels.push(epoch + 1);
            valLossChartInstance.data.datasets[0].data = [...valLosses];
            valLossChartInstance.update('none');

            lastChartUpdate = now;
          }

          showStatus(
            `Epoch ${epoch + 1}/${epochs} | Loss: ${logs.loss.toFixed(4)} | Val Loss: ${logs.val_loss.toFixed(4)}`,
            'info'
          );

          // Yield to browser to keep UI responsive
          await new Promise(r => setTimeout(r, 0));

          if (stopTraining) {
            model.stopTraining = true;
          }
        }
      }
    });

    showStatus('Training completed successfully!', 'success');

    // Cleanup
    xsTensor.dispose();
    ysTensor.dispose();
    model.dispose();

  } catch (error) {
    showStatus(`Training error: ${error.message}`, 'error');
  } finally {
    startTrainBtn.disabled = false;
    stopTrainBtn.disabled = true;
  }
};

/**
 * Handle Stop Training button
 */
stopTrainBtn.onclick = () => {
  stopTraining = true;
  showStatus('Training stopped by user', 'info');
};

// ============================================================================
// NETWORK ARCHITECTURE MODAL
// ============================================================================

showArchBtn.onclick = () => {
  const units = lstmUnitsInput.value;
  archText.textContent = `
╔════════════════════════════════════════════════════════════════════════╗
║           LSTM NETWORK ARCHITECTURE FOR TIME SERIES PREDICTION         ║
╚════════════════════════════════════════════════════════════════════════╝

INPUT LAYER
━━━━━━━━━━
  Shape: (batch_size, sequence_length, 1)
  Description: Normalized sine-wave sequences with sequence_length timesteps

      ↓

LSTM LAYER (Long Short-Term Memory)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Units: ${units}
  Activation: ReLU (Rectified Linear Unit)
  Return Sequences: False (returns only last output)
  Purpose: Captures temporal dependencies and patterns in the data
  
  Internal Gates:
    • Input Gate: Decides what information to store
    • Forget Gate: Decides what information to discard
    • Output Gate: Decides what to output from cell state

      ↓

DROPOUT LAYER
━━━━━━━━━━━━
  Rate: 0.2 (20%)
  Purpose: Regularization to prevent overfitting
  Effect: Randomly deactivates 20% of neurons during training

      ↓

DENSE HIDDEN LAYER
━━━━━━━━━━━━━━━━━
  Units: 32
  Activation: ReLU
  Purpose: Feature transformation and abstraction

      ↓

OUTPUT LAYER (Dense)
━━━━━━━━━━━━━━━━━━
  Units: 1
  Activation: Linear (identity)
  Purpose: Outputs the predicted next value in the sequence

═════════════════════════════════════════════════════════════════════════

TRAINING CONFIGURATION
━━━━━━━━━━━━━━━━━━━━━
  Loss Function: Mean Squared Error (MSE)
  Optimizer: Adam (adaptive learning rate)
  Metrics: Mean Absolute Error (MAE)

═════════════════════════════════════════════════════════════════════════
`;

  archModal.style.display = 'flex';
};

// Close modal when close button is clicked
document.addEventListener('DOMContentLoaded', () => {
  const modalCloseBtn = document.querySelector('.modal-close');
  if (modalCloseBtn) {
    modalCloseBtn.onclick = () => {
      archModal.style.display = 'none';
    };
  }

  // Close modal when clicking outside
  archModal.onclick = (e) => {
    if (e.target === archModal) {
      archModal.style.display = 'none';
    }
  };
});
