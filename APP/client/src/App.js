// SpamDetector.jsx
import React, { useState, useEffect } from 'react';
import { 
  TextField, 
  Button, 
  Card, 
  CardContent, 
  Typography, 
  Chip,
  Snackbar,
  CircularProgress,
  Tabs,
  Tab,
  Box
} from '@mui/material';
import axios from 'axios';


function SpamDetector() {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [snackbar, setSnackbar] = useState({ open: false, message: '' });
  const [tabValue, setTabValue] = useState(0);
  const [cacheStats, setCacheStats] = useState({ hits: 0, misses: 0 });

  const checkSpam = async () => {
    if (!text.trim()) return;
    
    setLoading(true);
    try {
      const response = await axios.post('http://localhost:6068/api/predict', { text });
      setResult(response.data);
      setCacheStats(prev => ({
        hits: prev.hits + (response.data.cache_hit ? 1 : 0),
        misses: prev.misses + (response.data.cache_hit ? 0 : 1)
      }));
    } catch (error) {
      setSnackbar({ open: true, message: 'Error analyzing text' });
    } finally {
      setLoading(false);
    }
  };

  const submitFeedback = async (userCorrection) => {
    try {
      const response = await axios.post('http://localhost:6068/api/feedback', {
        text: text.trim(),
        prediction: result.prediction,
        correction: userCorrection
      });
      
      setSnackbar({ 
        open: true, 
        message: response.data.status === 'success' 
          ? 'Thanks for your feedback!' 
          : 'Feedback submitted but with issues'
      });
    } catch (error) {
      setSnackbar({
        open: true,
        message: error.response?.data?.message || 
                'Failed to submit feedback'
      });
    }
};
  return (
    <div className="container">
      <Card className="detector-card">
        <CardContent>
          <Typography variant="h5" gutterBottom>
            Spam Message Detector
          </Typography>
          
          <Tabs value={tabValue} onChange={(e, newValue) => setTabValue(newValue)}>
            <Tab label="Detector" />
            <Tab label="Cache Stats" />
          </Tabs>
          
          {tabValue === 0 ? (
            <Box>
              <TextField
                label="Enter message"
                multiline
                rows={4}
                fullWidth
                variant="outlined"
                value={text}
                onChange={(e) => setText(e.target.value)}
                margin="normal"
              />
              
              <Button 
                variant="contained" 
                color="primary" 
                onClick={checkSpam}
                disabled={loading}
                startIcon={loading ? <CircularProgress size={20} /> : null}
              >
                Check for Spam
              </Button>
              
              {result && (
                <div className="result-container">
                  <Typography variant="h6" style={{ marginTop: 20 }}>
                    Result: 
                    <Chip 
                      label={result.prediction.toUpperCase()} 
                      color={result.prediction === 'spam' ? 'error' : 'success'}
                      style={{ marginLeft: 10 }}
                    />
                  </Typography>
                  
                  <Typography>
                    Confidence: {(result.confidence * 100).toFixed(2)}%
                  </Typography>
                  
                  <div className="probability-meter">
                    <div 
                      className="probability-bar ham"
                      style={{ width: `${result.details.probabilities.ham * 100}%` }}
                    >
                      HAM: {(result.details.probabilities.ham * 100).toFixed(1)}%
                    </div>
                    <div 
                      className="probability-bar spam"
                      style={{ width: `${result.details.probabilities.spam * 100}%` }}
                    >
                      SPAM: {(result.details.probabilities.spam * 100).toFixed(1)}%
                    </div>
                  </div>
                  
                  {result.cache_hit && (
                    <Typography variant="caption" color="textSecondary">
                      Served from cache
                    </Typography>
                  )}
                  
                  <div className="feedback-buttons">
                    <Typography>Was this correct?</Typography>
                    <Button 
                      size="small" 
                      onClick={() => submitFeedback(result.prediction)}
                    >
                      ✓ Yes
                    </Button>
                    <Button 
                      size="small" 
                      onClick={() => submitFeedback(
                        result.prediction === 'spam' ? 'ham' : 'spam'
                      )}
                    >
                      ✗ No
                    </Button>
                  </div>
                </div>
              )}
            </Box>
          ) : (
            <Box style={{ marginTop: 20 }}>
              <Typography variant="h6">Cache Performance</Typography>
              <Typography>Hits: {cacheStats.hits}</Typography>
              <Typography>Misses: {cacheStats.misses}</Typography>
              <Typography>
                Hit Rate: {cacheStats.hits + cacheStats.misses > 0 
                  ? ((cacheStats.hits / (cacheStats.hits + cacheStats.misses)) * 100).toFixed(1)
                  : 0}%
              </Typography>
            </Box>
          )}
        </CardContent>
      </Card>
      
      <Snackbar
        open={snackbar.open}
        autoHideDuration={3000}
        onClose={() => setSnackbar({ ...snackbar, open: false })}
        message={snackbar.message}
      />
    </div>
    
  );
}

export default SpamDetector;