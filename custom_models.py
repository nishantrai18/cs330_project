import learn2learn as l2l


Features = 'fets'
Inputs = 'input'


class CustomOmniglotFC(l2l.vision.models.OmniglotFC):

    def __init__(self, input_size, output_size, sizes=None, return_fets=None):
        super(CustomOmniglotFC, self).__init__(input_size, output_size, sizes)

        self.return_fets = return_fets

    def forward(self, x):
        B = x.shape[0]
        fets = self.features(x)
        fets.retain_grad()
        logits = self.classifier(fets)
        if self.return_fets == Features:
            return logits, fets
        elif self.return_fets == Inputs:
            return logits, x.view(B, -1)
        else:
            return logits, logits


class CustomMiniImagenetCNN(l2l.vision.models.MiniImagenetCNN):

    def __init__(self, output_size, hidden_size=32, layers=4, return_fets=False):
        super(CustomMiniImagenetCNN, self).__init__(output_size, hidden_size, layers)

        self.return_fets = return_fets

    def forward(self, x):
        B = x.shape[0]
        fets = self.features(x)
        fets.retain_grad()
        logits = self.classifier(fets)
        if self.return_fets == Features:
            return logits, fets
        elif self.return_fets == Inputs:
            return logits, x.view(B, -1)
        else:
            return logits, logits
