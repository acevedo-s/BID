def MLE_shells(idx,rs,probs):
  return (rs[idx]+1) * (probs[idx] + probs[idx+1]) / probs[idx] - 1
