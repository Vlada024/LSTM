/**
 * Main Application Module
 * Handles UI interactions and dataset generation with viewer
 */

class LSTMApp {
    constructor() {
        this.currentDataset = null;
        this.currentSampleIndex = 0;
        this.currentStartIndex = 0;
        this.overlayEnabled = false;
        this.previousSampleData = null;
        this.lossChartInstance = null;
        this.metricsChartInstance = null;
        this.stopTraining = false;
        this.initializeElements();
        this.attachEventListeners();
    }

    /**
     * Extract a scalar metric value from the model.fit() history object.
     * Handles different return shapes across TF.js versions (history.history arrays or tensor values).
     */
    _extractHistoryMetric(historyObj, key) {
        try {
            // Prefer historyObj.history (standard TF.js History)
            if (historyObj && historyObj.history && key in historyObj.history) {
                const arr = historyObj.history[key];
                if (Array.isArray(arr)) return arr[arr.length - 1];
                return arr;
            }

            // Fallback: direct field (older/alternate shapes)
            const val = historyObj && historyObj[key];
            if (val == null) return undefined;

            // If it's a tensor-like object, extract scalar
            if (typeof val.dataSync === 'function') return val.dataSync()[0];

            // If it's an array, take last element
            if (Array.isArray(val)) return val[val.length - 1];

            return val;
        } catch (e) {
            return undefined;
        }
    }

    /**
     * Render preview highlighting the current sample index.
     * Draws all samples faded and the selected sample prominently.
     */
    renderPreviewSample() {
        if (!this.currentDataset) return;

        const ctx = this.previewChart.getContext('2d');
        const width = this.previewChart.offsetWidth;
        const height = this.previewChart.offsetHeight || 300;

        this.previewChart.width = width;
        this.previewChart.height = height;

        const { sequences, stats } = this.currentDataset;

        // Background
        ctx.fillStyle = '#f8f9fa';
        ctx.fillRect(0, 0, width, height);

        // Grid
        ctx.strokeStyle = '#e0e0e0';
        ctx.lineWidth = 1;
        const gridSpacing = 50;
        for (let x = 0; x < width; x += gridSpacing) {
            ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke();
        }
        for (let y = 0; y < height; y += gridSpacing) {
            ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
        }

        const seqLength = sequences[0].length;
        const xScale = width / seqLength;
        const yScale = (height * 0.8) / (stats.max - stats.min);
        const yOffset = height / 2;

        // Draw all sequences faded
        ctx.lineWidth = 1.2;
        ctx.strokeStyle = 'rgba(100,100,100,0.15)';
        for (let i = 0; i < sequences.length; i++) {
            const seq = sequences[i];
            ctx.beginPath();
            for (let t = 0; t < seq.length; t++) {
                const x = t * xScale;
                const y = yOffset - (seq[t] - stats.mean) * yScale;
                if (t === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            }
            ctx.stroke();
        }

        // Highlight selected sample
        const idx = Math.max(0, Math.min(this.currentSampleIndex || 0, sequences.length - 1));
        const selected = sequences[idx];
        ctx.strokeStyle = '#ff6b6b';
        ctx.lineWidth = 3.0;
        ctx.beginPath();
        for (let t = 0; t < selected.length; t++) {
            const x = t * xScale;
            const y = yOffset - (selected[t] - stats.mean) * yScale;
            if (t === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Update label
        if (this.previewSampleLabel) {
            this.previewSampleLabel.textContent = `Sample: ${idx} / ${sequences.length - 1}`;
        }
    }

    goToPreviousPreview() {
        if (!this.currentDataset) return;
        this.currentSampleIndex = Math.max(0, (this.currentSampleIndex || 0) - 1);
        this.renderPreviewSample();
    }

    goToNextPreview() {
        if (!this.currentDataset) return;
        const maxIndex = this.currentDataset.sequences.length - 1;
        this.currentSampleIndex = Math.min(maxIndex, (this.currentSampleIndex || 0) + 1);
        this.renderPreviewSample();
    }

    /**
     * Initialize DOM elements
     */
    initializeElements() {
        // Input fields
        this.samplesInput = document.getElementById('samples');
        this.sequenceLengthInput = document.getElementById('sequence-length');
        this.seedInput = document.getElementById('seed');
        this.noiseStdInput = document.getElementById('noise-std');
        this.ampMinInput = document.getElementById('amp-min');
        this.ampMaxInput = document.getElementById('amp-max');
        this.freqMinInput = document.getElementById('freq-min');
        this.freqMaxInput = document.getElementById('freq-max');

        // Buttons
        this.generateBtn = document.getElementById('generate-btn');
        this.exportBtn = document.getElementById('export-btn');

        // Status and display
        this.statusDiv = document.getElementById('status');
        this.previewSection = document.getElementById('preview-section');
        this.previewChart = document.getElementById('preview-chart');
        this.previewControls = document.getElementById('preview-controls');
        this.previewPrevBtn = document.getElementById('preview-prev-btn');
        this.previewNextBtn = document.getElementById('preview-next-btn');
        this.previewRenderBtn = document.getElementById('preview-render-btn');
        this.previewSampleLabel = document.getElementById('preview-sample-label');
        this.statsDiv = document.getElementById('stats');

        // Training elements
        this.trainingSection = document.getElementById('training-section');
        this.unitsInput = document.getElementById('units');
        this.batchSizeInput = document.getElementById('batch-size');
        this.epochsInput = document.getElementById('epochs');
        this.learningRateInput = document.getElementById('learning-rate');
        this.validationSplitInput = document.getElementById('validation-split');
        this.lowCpuCheckbox = document.getElementById('low-cpu');
        this.trainSubsetInput = document.getElementById('train-subset');
        this.startTrainingBtn = document.getElementById('start-training-btn');
        this.stopTrainingBtn = document.getElementById('stop-training-btn');
        this.trainingStatus = document.getElementById('training-status');
        this.trainingProgress = document.getElementById('training-progress');
        this.progressFill = document.getElementById('progress-fill');
        this.epochInfo = document.getElementById('epoch-info');
        this.lossInfo = document.getElementById('loss-info');
        this.trainingChartsContainer = document.getElementById('training-charts-container');
        this.trainingLossChart = document.getElementById('training-loss-chart');
        this.trainingMetricsChart = document.getElementById('training-metrics-chart');
        this.trainingMetrics = document.getElementById('training-metrics');

        // Architecture modal elements
        this.architectureBtn = document.getElementById('architecture-btn');
        this.architectureModal = document.getElementById('architecture-modal');
        this.modalClose = document.querySelector('.modal-close');
        this.architectureAscii = document.getElementById('architecture-ascii');
    }

    /**
     * Attach event listeners
     */
    attachEventListeners() {
        this.generateBtn.addEventListener('click', () => this.handleGenerate());
        this.exportBtn.addEventListener('click', () => this.handleExport());

        // Preview controls
        if (this.previewPrevBtn && this.previewNextBtn && this.previewRenderBtn) {
            this.previewPrevBtn.addEventListener('click', () => this.goToPreviousPreview());
            this.previewNextBtn.addEventListener('click', () => this.goToNextPreview());
            this.previewRenderBtn.addEventListener('click', () => this.renderPreviewSample());
        }

        // Training controls (guarded)
        if (this.startTrainingBtn) {
            this.startTrainingBtn.addEventListener('click', () => this.handleStartTraining());
        }
        if (this.stopTrainingBtn) {
            this.stopTrainingBtn.addEventListener('click', () => this.handleStopTraining());
        }

        // Architecture modal controls
        this.architectureBtn.addEventListener('click', () => this.showArchitecture());
        this.modalClose.addEventListener('click', () => this.closeArchitecture());
        this.architectureModal.addEventListener('click', (e) => {
            if (e.target === this.architectureModal) {
                this.closeArchitecture();
            }
        });
    }

    /**
     * Handle dataset generation
     */
    async handleGenerate() {
        try {
            this.setStatus('Generating dataset...', 'info');
            this.generateBtn.disabled = true;

            // Get configuration from inputs
            const config = {
                samples: parseInt(this.samplesInput.value, 10),
                sequenceLength: parseInt(this.sequenceLengthInput.value, 10),
                seed: parseInt(this.seedInput.value, 10),
                noiseStd: parseFloat(this.noiseStdInput.value),
                ampMin: parseFloat(this.ampMinInput.value),
                ampMax: parseFloat(this.ampMaxInput.value),
                freqMin: parseFloat(this.freqMinInput.value),
                freqMax: parseFloat(this.freqMaxInput.value)
            };

            // Validate configuration
            this.validateConfig(config);

            // Generate dataset
            const startTime = performance.now();
            this.currentDataset = DatasetGenerator.generate(config);
            const endTime = performance.now();

            // Display success message
            const duration = ((endTime - startTime) / 1000).toFixed(2);
            this.setStatus(
                `✓ Dataset generated successfully in ${duration}s | ${config.samples} samples, ${config.sequenceLength} time steps`,
                'success'
            );

            // Display preview and statistics
            // start with sample 0 and enable controls
            this.currentSampleIndex = 0;
            this.previewControls && (this.previewControls.style.display = 'flex');
            this.renderPreviewSample();
            this.displayStats();

            // Ensure training controls are enabled
            this.trainingSection.style.display = 'block';
            if (this.startTrainingBtn) this.startTrainingBtn.disabled = false;
            if (this.stopTrainingBtn) this.stopTrainingBtn.disabled = true;

            // Show training section
            this.trainingSection.style.display = 'block';

            // Enable export button
            this.exportBtn.disabled = false;
            this.generateBtn.disabled = false;

        } catch (error) {
            this.setStatus(`✗ Error: ${error.message}`, 'error');
            this.generateBtn.disabled = false;
        }
    }

    /**
     * Validate configuration parameters
     */
    validateConfig(config) {
        if (config.samples < 1) throw new Error('Samples must be at least 1');
        if (config.sequenceLength < 1) throw new Error('Sequence length must be at least 1');
        if (config.noiseStd < 0) throw new Error('Noise std must be non-negative');
        if (config.ampMin >= config.ampMax) throw new Error('Amplitude min must be less than max');
        if (config.freqMin >= config.freqMax) throw new Error('Frequency min must be less than max');
        if (config.ampMin < 0) throw new Error('Amplitude must be positive');
        if (config.freqMin < 0) throw new Error('Frequency must be positive');
    }

    /**
     * Display dataset preview with chart
     */
    displayPreview() {
        this.previewSection.style.display = 'block';

        // Draw multiple sample sequences
        const ctx = this.previewChart.getContext('2d');
        const width = this.previewChart.offsetWidth;
        const height = this.previewChart.offsetHeight || 300;

        this.previewChart.width = width;
        this.previewChart.height = height;

        const { sequences, targets, stats } = this.currentDataset;

        // Draw background
        ctx.fillStyle = '#f8f9fa';
        ctx.fillRect(0, 0, width, height);

        // Draw grid
        ctx.strokeStyle = '#e0e0e0';
        ctx.lineWidth = 1;
        const gridSpacing = 50;
        for (let x = 0; x < width; x += gridSpacing) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, height);
            ctx.stroke();
        }
        for (let y = 0; y < height; y += gridSpacing) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();
        }

        // Calculate scaling factors
        const seqLength = sequences[0].length;
        const xScale = width / seqLength;
        const yScale = (height * 0.8) / (stats.max - stats.min);
        const yOffset = height / 2;

        // Draw up to 5 sample sequences
        const samplesToDraw = Math.min(5, sequences.length);
        const colors = [
            '#667eea',
            '#764ba2',
            '#f093fb',
            '#4facfe',
            '#00f2fe'
        ];

        for (let sampleIdx = 0; sampleIdx < samplesToDraw; sampleIdx++) {
            const sequence = sequences[sampleIdx];
            ctx.strokeStyle = colors[sampleIdx];
            ctx.lineWidth = 2;
            ctx.beginPath();

            for (let t = 0; t < sequence.length; t++) {
                const x = t * xScale;
                const y = yOffset - (sequence[t] - stats.mean) * yScale;

                if (t === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            }
            ctx.stroke();
        }

        // Draw axes
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(0, yOffset);
        ctx.lineTo(width, yOffset);
        ctx.stroke();

        // Add labels
        ctx.fillStyle = '#666';
        ctx.font = '12px sans-serif';
        ctx.fillText('Time Steps', width - 80, height - 10);
        ctx.fillText('Amplitude', 10, 20);
    }

    /**
     * Display dataset statistics
     */
    displayStats() {
        const { stats } = this.currentDataset;

        this.statsDiv.innerHTML = `
            <div class="stat-box">
                <div class="stat-label">Total Samples</div>
                <div class="stat-value">${stats.samplesCount}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Sequence Length</div>
                <div class="stat-value">${stats.sequenceLength}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Data Min</div>
                <div class="stat-value">${stats.min.toFixed(3)}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Data Max</div>
                <div class="stat-value">${stats.max.toFixed(3)}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Mean</div>
                <div class="stat-value">${stats.mean.toFixed(3)}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Std Dev</div>
                <div class="stat-value">${stats.std.toFixed(3)}</div>
            </div>
        `;
    }

    /**
     * Handle dataset export
     */
    handleExport() {
        try {
            if (!this.currentDataset) {
                this.setStatus('✗ No dataset to export', 'error');
                return;
            }

            const timestamp = new Date().toISOString().slice(0, 10);
            const filename = `lstm_dataset_${timestamp}.json`;

            DatasetGenerator.exportJSON(this.currentDataset, filename);
            this.setStatus(`✓ Dataset exported as ${filename}`, 'success');

        } catch (error) {
            this.setStatus(`✗ Export error: ${error.message}`, 'error');
        }
    }

    /**
     * Display status message
     */
    setStatus(message, type = 'info') {
        this.statusDiv.textContent = message;
        this.statusDiv.className = `status ${type}`;

        // Auto-hide info and success messages after 5 seconds
        if (type !== 'error') {
            setTimeout(() => {
                this.statusDiv.className = 'status';
            }, 5000);
        }
    }

    /**
     * Handle training start
     */
    async handleStartTraining() {
        try {
            if (!this.currentDataset) {
                this.trainingStatus.textContent = '✗ Please generate a dataset first';
                this.trainingStatus.className = 'status error';
                return;
            }

            // Basic validation of training config
            const units = parseInt(this.unitsInput.value, 10);
            const batchSize = parseInt(this.batchSizeInput.value, 10);
            const epochs = parseInt(this.epochsInput.value, 10);
            const learningRate = parseFloat(this.learningRateInput.value);
            const validationSplit = parseFloat(this.validationSplitInput.value);

            if (Number.isNaN(units) || units <= 0) throw new Error('Invalid LSTM units');
            if (Number.isNaN(batchSize) || batchSize <= 0) throw new Error('Invalid batch size');
            if (Number.isNaN(epochs) || epochs <= 0) throw new Error('Invalid epochs');
            if (Number.isNaN(learningRate) || learningRate <= 0) throw new Error('Invalid learning rate');
            if (Number.isNaN(validationSplit) || validationSplit < 0 || validationSplit >= 1) throw new Error('Invalid validation split');

            this.startTrainingBtn.disabled = true;
            this.stopTrainingBtn.disabled = false;

            const config = {
                units: parseInt(this.unitsInput.value, 10),
                batchSize: parseInt(this.batchSizeInput.value, 10),
                epochs: parseInt(this.epochsInput.value, 10),
                learningRate: parseFloat(this.learningRateInput.value),
                validationSplit: parseFloat(this.validationSplitInput.value),
                lowCpu: !!(this.lowCpuCheckbox && this.lowCpuCheckbox.checked),
                subsetPercent: this.trainSubsetInput ? parseInt(this.trainSubsetInput.value, 10) : 20
            };

            await this.trainModel(config);

        } catch (error) {
            this.trainingStatus.textContent = `✗ Training error: ${error.message}`;
            this.trainingStatus.className = 'status error';
            this.startTrainingBtn.disabled = false;
            this.stopTrainingBtn.disabled = true;
        }
    }

    /**
     * Handle training stop
     */
    handleStopTraining() {
        this.stopTraining = true;
        this.stopTrainingBtn.disabled = true;
    }

    /**
     * Train LSTM model with TensorFlow.js
     */
    async trainModel(config) {
        try {
            this.trainingStatus.textContent = 'Preparing data...';
            this.trainingStatus.className = 'status info';
            this.trainingProgress.style.display = 'block';
            this.trainingChartsContainer.style.display = 'grid';
            this.trainingMetrics.style.display = 'grid';
            
            this.stopTraining = false;
            const { sequences: allSequences, targets: allTargets } = this.currentDataset;

            // Optionally use a subset for low-CPU mode
            let sequences = allSequences;
            let targets = allTargets;
            if (config.lowCpu) {
                const pct = Math.min(100, Math.max(1, parseInt(config.subsetPercent || 20, 10)));
                const subsetCount = Math.max(1, Math.floor(allSequences.length * (pct / 100)));
                // Random shuffle indices and take subsetCount
                const indices = Array.from({ length: allSequences.length }, (_, i) => i);
                for (let i = indices.length - 1; i > 0; i--) {
                    const j = Math.floor(Math.random() * (i + 1));
                    [indices[i], indices[j]] = [indices[j], indices[i]];
                }
                const chosen = indices.slice(0, subsetCount);
                sequences = chosen.map(i => allSequences[i]);
                targets = chosen.map(i => allTargets[i]);
                this.trainingStatus.textContent = `Low-CPU mode: training on ${subsetCount}/${allSequences.length} samples`;
                this.trainingStatus.className = 'status info';
            }

            // Prepare data (normalize & reshape)
            const xs = sequences.map(seq => {
                const normalized = seq.map(v => (v - this.currentDataset.stats.mean) / this.currentDataset.stats.std);
                // Reshape to [timesteps, 1] for LSTM
                return normalized.map(val => [val]);
            });

            const ys = targets.map(t => (t - this.currentDataset.stats.mean) / this.currentDataset.stats.std);

            // Convert to tensors - xs shape: [samples, timesteps, features]
            const xsTensor = tf.tensor3d(xs);
            const ysTensor = tf.tensor2d(ys, [ys.length, 1]);

            // Build model
            const model = tf.sequential({
                layers: [
                    tf.layers.lstm({
                        units: config.units,
                        returnSequences: false,
                        inputShape: [sequences[0].length, 1],
                        activation: 'relu'
                    }),
                    tf.layers.dropout({ rate: 0.2 }),
                    tf.layers.dense({ units: 32, activation: 'relu' }),
                    tf.layers.dense({ units: 1, activation: 'linear' })
                ]
            });

            model.compile({
                optimizer: tf.train.adam(config.learningRate),
                loss: 'meanSquaredError',
                metrics: ['mae']
            });

            // Training history
            const trainLosses = [];
            const valLosses = [];
            const trainMAEs = [];
            const valMAEs = [];

            // Train model
            const startTime = Date.now();
            for (let epoch = 0; epoch < config.epochs && !this.stopTraining; epoch++) {
                const history = await model.fit(xsTensor, ysTensor, {
                    epochs: 1,
                    batchSize: config.batchSize,
                    validationSplit: config.validationSplit,
                    verbose: 0,
                    shuffle: true
                });

                    // Extract scalar values robustly from history (handles different TF.js shapes)
                    let trainLoss = this._extractHistoryMetric(history, 'loss');
                    let valLoss = this._extractHistoryMetric(history, 'val_loss');
                    let trainMAE = this._extractHistoryMetric(history, 'mae');
                    let valMAE = this._extractHistoryMetric(history, 'val_mae');

                    // Fallback names (some TF.js versions use different keys)
                    if (valLoss === undefined) valLoss = this._extractHistoryMetric(history, 'valLoss');
                    if (trainMAE === undefined) trainMAE = this._extractHistoryMetric(history, 'meanAbsoluteError') || this._extractHistoryMetric(history, 'mae');
                    if (valMAE === undefined) valMAE = this._extractHistoryMetric(history, 'valMeanAbsoluteError') || this._extractHistoryMetric(history, 'val_mae');

                    // Ensure numbers
                    trainLoss = typeof trainLoss === 'number' ? trainLoss : (Array.isArray(trainLoss) ? trainLoss[trainLoss.length - 1] : Number(trainLoss) || 0);
                    valLoss = typeof valLoss === 'number' ? valLoss : (Array.isArray(valLoss) ? valLoss[valLoss.length - 1] : Number(valLoss) || 0);
                    trainMAE = typeof trainMAE === 'number' ? trainMAE : (Array.isArray(trainMAE) ? trainMAE[trainMAE.length - 1] : Number(trainMAE) || 0);
                    valMAE = typeof valMAE === 'number' ? valMAE : (Array.isArray(valMAE) ? valMAE[valMAE.length - 1] : Number(valMAE) || 0);

                    trainLosses.push(trainLoss);
                    valLosses.push(valLoss);
                    trainMAEs.push(trainMAE);
                    valMAEs.push(valMAE);

                // Update UI
                const progress = ((epoch + 1) / config.epochs) * 100;
                this.progressFill.style.width = progress + '%';
                this.epochInfo.textContent = `Epoch ${epoch + 1}/${config.epochs}`;
                    this.lossInfo.textContent = `Loss: ${trainLoss.toFixed(4)} | Val Loss: ${valLoss.toFixed(4)}`;

                // Update charts
                this.updateTrainingCharts(trainLosses, valLosses, trainMAEs, valMAEs);

                    // Yield to browser and underlying TF backends so UI stays responsive
                    if (typeof tf !== 'undefined' && tf.nextFrame) {
                        await tf.nextFrame();
                    } else {
                        await new Promise(resolve => setTimeout(resolve, 50));
                    }
            }

            const duration = ((Date.now() - startTime) / 1000).toFixed(2);

            if (this.stopTraining) {
                this.trainingStatus.textContent = `⊘ Training stopped after ${duration}s`;
                this.trainingStatus.className = 'status warning';
            } else {
                this.trainingStatus.textContent = `✓ Training completed in ${duration}s`;
                this.trainingStatus.className = 'status success';
            }

            // Display final metrics
            this.displayTrainingMetrics(trainLosses, valLosses, trainMAEs, valMAEs);

            // Cleanup
            xsTensor.dispose();
            ysTensor.dispose();
            model.dispose();

            this.startTrainingBtn.disabled = false;
            this.stopTrainingBtn.disabled = true;

        } catch (error) {
            throw error;
        }
    }

    /**
     * Update dual training charts
     */
    updateTrainingCharts(trainLosses, valLosses, trainMAEs, valMAEs) {
        this.trainingChartsContainer.style.display = 'grid';
        
        const epochLabels = Array.from({ length: trainLosses.length }, (_, i) => i + 1);

        // Loss Chart
        if (this.lossChartInstance) {
            this.lossChartInstance.data.labels = epochLabels;
            this.lossChartInstance.data.datasets[0].data = trainLosses;
            this.lossChartInstance.data.datasets[1].data = valLosses;
            this.lossChartInstance.update('none');
        } else {
            const lossCtx = this.trainingLossChart.getContext('2d');
            this.lossChartInstance = new Chart(lossCtx, {
                type: 'line',
                data: {
                    labels: epochLabels,
                    datasets: [
                        {
                            label: 'Training Loss',
                            data: trainLosses,
                            borderColor: '#667eea',
                            backgroundColor: 'rgba(102, 126, 234, 0.15)',
                            borderWidth: 2.5,
                            tension: 0.4,
                            fill: true,
                            pointRadius: 4,
                            pointHoverRadius: 6,
                            pointBackgroundColor: '#667eea',
                            pointBorderColor: '#fff',
                            pointBorderWidth: 2
                        },
                        {
                            label: 'Validation Loss',
                            data: valLosses,
                            borderColor: '#e74c3c',
                            backgroundColor: 'rgba(231, 76, 60, 0.15)',
                            borderWidth: 2.5,
                            tension: 0.4,
                            fill: true,
                            pointRadius: 4,
                            pointHoverRadius: 6,
                            pointBackgroundColor: '#e74c3c',
                            pointBorderColor: '#fff',
                            pointBorderWidth: 2
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    interaction: {
                        mode: 'index',
                        intersect: false
                    },
                    plugins: {
                        legend: {
                            display: true,
                            position: 'top',
                            labels: {
                                usePointStyle: true,
                                padding: 15,
                                font: { weight: '600', size: 12 }
                            }
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            grid: { color: 'rgba(0,0,0,0.05)' },
                            ticks: { font: { size: 11 } },
                            title: { display: true, text: 'Loss (MSE)', font: { weight: '600' } }
                        },
                        x: {
                            grid: { display: false },
                            ticks: { font: { size: 11 } },
                            title: { display: true, text: 'Epoch', font: { weight: '600' } }
                        }
                    }
                }
            });
        }

        // Metrics Chart (MAE)
        if (this.metricsChartInstance) {
            this.metricsChartInstance.data.labels = epochLabels;
            this.metricsChartInstance.data.datasets[0].data = trainMAEs;
            this.metricsChartInstance.data.datasets[1].data = valMAEs;
            this.metricsChartInstance.update('none');
        } else {
            const metricsCtx = this.trainingMetricsChart.getContext('2d');
            this.metricsChartInstance = new Chart(metricsCtx, {
                type: 'line',
                data: {
                    labels: epochLabels,
                    datasets: [
                        {
                            label: 'Training MAE',
                            data: trainMAEs,
                            borderColor: '#3498db',
                            backgroundColor: 'rgba(52, 152, 219, 0.15)',
                            borderWidth: 2.5,
                            tension: 0.4,
                            fill: true,
                            pointRadius: 4,
                            pointHoverRadius: 6,
                            pointBackgroundColor: '#3498db',
                            pointBorderColor: '#fff',
                            pointBorderWidth: 2
                        },
                        {
                            label: 'Validation MAE',
                            data: valMAEs,
                            borderColor: '#f39c12',
                            backgroundColor: 'rgba(243, 156, 18, 0.15)',
                            borderWidth: 2.5,
                            tension: 0.4,
                            fill: true,
                            pointRadius: 4,
                            pointHoverRadius: 6,
                            pointBackgroundColor: '#f39c12',
                            pointBorderColor: '#fff',
                            pointBorderWidth: 2
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    interaction: {
                        mode: 'index',
                        intersect: false
                    },
                    plugins: {
                        legend: {
                            display: true,
                            position: 'top',
                            labels: {
                                usePointStyle: true,
                                padding: 15,
                                font: { weight: '600', size: 12 }
                            }
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            grid: { color: 'rgba(0,0,0,0.05)' },
                            ticks: { font: { size: 11 } },
                            title: { display: true, text: 'Mean Absolute Error', font: { weight: '600' } }
                        },
                        x: {
                            grid: { display: false },
                            ticks: { font: { size: 11 } },
                            title: { display: true, text: 'Epoch', font: { weight: '600' } }
                        }
                    }
                }
            });
        }
    }

    /**
     * Display final training metrics
     */
    displayTrainingMetrics(trainLosses, valLosses, trainMAEs, valMAEs) {
        const finalTrainLoss = trainLosses[trainLosses.length - 1];
        const finalValLoss = valLosses[valLosses.length - 1];
        const bestValLoss = Math.min(...valLosses);
        const bestEpoch = valLosses.indexOf(bestValLoss) + 1;

        const html = `
            <div class="metric-box">
                <div class="metric-label">Final Train Loss</div>
                <div class="metric-value">${finalTrainLoss.toFixed(6)}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Final Val Loss</div>
                <div class="metric-value">${finalValLoss.toFixed(6)}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Best Val Loss</div>
                <div class="metric-value success">${bestValLoss.toFixed(6)}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Best Epoch</div>
                <div class="metric-value">${bestEpoch}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Train/Val Ratio</div>
                <div class="metric-value ${finalValLoss > finalTrainLoss * 1.1 ? 'warning' : 'success'}">
                    ${(finalValLoss / finalTrainLoss).toFixed(2)}x
                </div>
            </div>
        `;
        this.trainingMetrics.innerHTML = html;
    }

    /**
     * Show network architecture modal
     */
    showArchitecture() {
        // Update architecture diagram with current units
        const units = parseInt(this.unitsInput.value, 10);
        const lr = parseFloat(this.learningRateInput.value);
        
        // Update dynamic values in the modal
        document.getElementById('arch-units').textContent = units;
        document.getElementById('arch-units-2').textContent = units;
        document.getElementById('arch-units-3').textContent = units;
        document.getElementById('arch-lr').textContent = lr.toFixed(4);

        // Generate ASCII diagram
        const seqLen = this.sequenceLengthInput.value;
        const diagram = this.generateArchitectureDiagram(seqLen, units);
        this.architectureAscii.textContent = diagram;

        this.architectureModal.style.display = 'flex';
    }

    /**
     * Close architecture modal
     */
    closeArchitecture() {
        this.architectureModal.style.display = 'none';
    }

    /**
     * Generate ASCII architecture diagram
     */
    generateArchitectureDiagram(seqLen, units) {
        const diagram = `
╔════════════════════════════════════════════════════════════════════════╗
║                    LSTM TIME SERIES PREDICTOR                         ║
╚════════════════════════════════════════════════════════════════════════╝

                        INPUT LAYER
                            │
                    ┌───────▼────────┐
                    │ Raw Sequences  │
                    │  Shape: (B, ${String(seqLen).padStart(3)}, 1)  │
                    │  Normalized    │
                    └───────┬────────┘
                            │
                    NORMALIZATION (Z-Score)
                            │
                    ┌───────▼────────────────────┐
                    │       LSTM LAYER           │
                    │  • Units: ${String(units).padStart(2)}              │
                    │  • Activation: ReLU        │
                    │  • Dropout: 20%            │
                    │  • Output: (B, ${String(units).padStart(2)})       │
                    └───────┬────────────────────┘
                            │
                    ┌───────▼────────┐
                    │ Dense Layer    │
                    │  • Units: 32   │
                    │  • Activation: │
                    │    ReLU        │
                    │  • Output:     │
                    │    (B, 32)     │
                    └───────┬────────┘
                            │
                    ┌───────▼────────┐
                    │ Output Layer   │
                    │  • Units: 1    │
                    │  • Activation: │
                    │    Linear      │
                    │  • Output:     │
                    │    (B, 1)      │
                    └───────┬────────┘
                            │
                        PREDICTION
                        (Next Value)

KEY COMPONENTS:

  INPUT      → Sequence of normalized values at ${seqLen} timesteps
  LSTM       → Learns long-term dependencies in temporal data
  DROPOUT    → Regularization (prevents overfitting)
  DENSE      → Feature transformation (32 hidden units)
  OUTPUT     → Predicts next value in the sequence

LOSS & OPTIMIZATION:

  Loss Function:  MSE (Mean Squared Error)
  Optimizer:      Adam (adaptive learning rate)
  Metrics:        MAE (Mean Absolute Error)

`;
        return diagram;
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new LSTMApp();
});
