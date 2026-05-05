import React, { useState } from 'react';
import CardioSoundDashboard from './components/CardioSoundDashboard';

const emptyConfidence = [
  { label: 'Normal', value: 0 },
  { label: 'Murmur', value: 0 },
  { label: 'Extrastole', value: 0 },
  { label: 'Artifact', value: 0 },
];

export default function App() {
  const [state, setState] = useState({
    prediction: 'Normal',
    confidence: 0,
    confidenceBreakdown: emptyConfidence,
    imageUrl: '',
    explanation: 'Upload a heart sound recording to generate the analysis output.',
    loading: false,
    error: '',
  });

  const handleAnalyze = async (file) => {
    if (!file) {
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    setState((current) => ({ ...current, loading: true, error: '' }));

    try {
      const response = await fetch('/api/analyze', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Analysis request failed');
      }

      setState({
        prediction: data.result || 'Normal',
        confidence: typeof data.confidence === 'number' ? data.confidence * 100 : 0,
        confidenceBreakdown: Array.isArray(data.confidence_breakdown) ? data.confidence_breakdown : emptyConfidence,
        imageUrl: data.image_url || '',
        explanation: data.explanation || 'Analysis completed.',
        loading: false,
        error: '',
      });
    } catch (error) {
      setState((current) => ({
        ...current,
        loading: false,
        error: error instanceof Error ? error.message : 'Unable to analyze file',
      }));
    }
  };

  return (
    <CardioSoundDashboard
      initialPrediction={state.prediction}
      initialConfidence={state.confidence}
      initialConfidenceBreakdown={state.confidenceBreakdown}
      initialGradcamImageUrl={state.imageUrl}
      initialExplanation={state.explanation}
      isLoading={state.loading}
      error={state.error}
      onAnalyze={handleAnalyze}
    />
  );
}
