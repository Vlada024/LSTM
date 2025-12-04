/**
 * Dataset Generator Module
 * Generates synthetic sine wave time series data with configurable parameters
 */

class DatasetGenerator {
    /**
     * Creates a seeded random number generator
     * @param {number} seed - The seed value
     * @returns {function} A random number generator function
     */
    static seededRandom(seed) {
        return function() {
            const x = Math.sin(seed++) * 10000;
            return x - Math.floor(x);
        };
    }

    /**
     * Generate synthetic sine wave dataset
     * @param {Object} config - Configuration object
     * @param {number} config.samples - Number of samples to generate
     * @param {number} config.sequenceLength - Length of each sequence
     * @param {number} config.seed - Random seed for reproducibility
     * @param {number} config.noiseStd - Standard deviation of Gaussian noise
     * @param {number} config.ampMin - Minimum amplitude
     * @param {number} config.ampMax - Maximum amplitude
     * @param {number} config.freqMin - Minimum frequency
     * @param {number} config.freqMax - Maximum frequency
     * @returns {Object} Dataset with sequences, targets, and metadata
     */
    static generate(config) {
        const {
            samples,
            sequenceLength,
            seed,
            noiseStd,
            ampMin,
            ampMax,
            freqMin,
            freqMax
        } = config;

        // Initialize seeded RNG
        const random = this.seededRandom(seed);

        // Container for sequences and targets
        const sequences = [];
        const targets = [];
        const metadata = [];

        // Generate samples
        for (let i = 0; i < samples; i++) {
            // Random parameters for this sample
            const amplitude = ampMin + random() * (ampMax - ampMin);
            const frequency = freqMin + random() * (freqMax - freqMin);
            const phase = random() * 2 * Math.PI;

            // Generate sequence
            const sequence = [];
            for (let t = 0; t < sequenceLength; t++) {
                // Base sine wave
                let value = amplitude * Math.sin(2 * Math.PI * frequency * t + phase);
                
                // Add Gaussian noise
                if (noiseStd > 0) {
                    value += this.gaussianRandom(random) * noiseStd;
                }

                sequence.push(value);
            }

            // Target: predict the next value
            const nextT = sequenceLength;
            const targetValue = amplitude * Math.sin(2 * Math.PI * frequency * nextT + phase);
            
            sequences.push(sequence);
            targets.push(targetValue);
            metadata.push({
                sampleId: i,
                amplitude,
                frequency,
                phase
            });
        }

        // Calculate statistics
        const allValues = sequences.flat();
        const stats = {
            min: Math.min(...allValues),
            max: Math.max(...allValues),
            mean: allValues.reduce((a, b) => a + b, 0) / allValues.length,
            std: this.calculateStd(allValues),
            samplesCount: samples,
            sequenceLength: sequenceLength,
            generatedAt: new Date().toISOString(),
            config: {
                seed,
                noiseStd,
                ampMin,
                ampMax,
                freqMin,
                freqMax
            }
        };

        return {
            sequences,
            targets,
            metadata,
            stats
        };
    }

    /**
     * Box-Muller transform to generate Gaussian random numbers
     * @param {function} random - Random number generator (0-1)
     * @returns {number} Gaussian random number
     */
    static gaussianRandom(random) {
        let u1, u2;
        do {
            u1 = random();
        } while (u1 <= 1e-10); // Avoid log(0)
        
        u2 = random();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }

    /**
     * Calculate standard deviation
     * @param {number[]} values - Array of values
     * @returns {number} Standard deviation
     */
    static calculateStd(values) {
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length;
        return Math.sqrt(variance);
    }

    /**
     * Export dataset to JSON
     * @param {Object} dataset - Dataset object from generate()
     * @param {string} filename - Output filename
     */
    static exportJSON(dataset, filename = 'dataset.json') {
        const json = JSON.stringify(dataset, null, 2);
        const blob = new Blob([json], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    }

    /**
     * Export dataset to CSV
     * @param {Object} dataset - Dataset object from generate()
     * @param {string} filename - Output filename
     */
    static exportCSV(dataset, filename = 'dataset.csv') {
        let csv = 'sample_id,time_step,value,target,amplitude,frequency,phase\n';

        const { sequences, targets, metadata } = dataset;

        for (let i = 0; i < sequences.length; i++) {
            const sequence = sequences[i];
            const target = targets[i];
            const meta = metadata[i];

            for (let t = 0; t < sequence.length; t++) {
                csv += `${i},${t},${sequence[t]},${target},${meta.amplitude},${meta.frequency},${meta.phase}\n`;
            }
        }

        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    }
}
