import numpy as np

class VotingClassifier:

    def __init__(self, models, voting='hard'):
        self.models = models
        self.voting = voting

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)

    def predict(self, X):

        all_predictions = []

        for model in self.models:
            preds = model.predict(X)
            all_predictions.append(preds)

        all_predictions = np.array(all_predictions)

        if self.voting == 'hard':

            final_predictions = []

            for i in range(X.shape[0]):

                votes = {}

                for j in range(len(self.models)):
                    label = all_predictions[j][i]

                    if label not in votes:
                        votes[label] = 1
                    else:
                        votes[label] += 1

                best_label = None
                max_votes = -1

                for label in votes:
                    if votes[label] > max_votes:
                        max_votes = votes[label]
                        best_label = label

                final_predictions.append(best_label)

            return np.array(final_predictions)


        elif self.voting == 'soft':

            all_probabilities = []

            for model in self.models:
                probs = model.predict_proba(X)
                all_probabilities.append(probs)

            all_probabilities = np.array(all_probabilities)

            avg_prob = np.mean(all_probabilities, axis=0)

            final_predictions = []

            for i in range(len(avg_prob)):

                best_class = 0
                best_value = avg_prob[i][0]

                for c in range(1, len(avg_prob[i])):
                    if avg_prob[i][c] > best_value:
                        best_value = avg_prob[i][c]
                        best_class = c

                final_predictions.append(best_class)

            return np.array(final_predictions)