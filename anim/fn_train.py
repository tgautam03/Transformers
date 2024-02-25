lr = 0.001
num_epochs = 20

model = Transformer(emb=32, heads=4, max_seq_length=max_length, vocab_size=len(vocab)).to("cuda")

opt = torch.optim.Adam(lr=lr, params=model.parameters())

for epoch in range(num_epochs):
    for batch in train_data_loader:
        opt.zero_grad()

        input = batch["id"].to("cuda")
        output = batch["label"].to("cuda")

        preds, _ = model(input)
        loss = F.nll_loss(preds, output)

        loss.backward()
        opt.step()

        with torch.no_grad():
            tot, cor= 0.0, 0.0
            for batch in test_data_loader:
                input = batch["id"].to("cuda")
                output = batch["label"].to("cuda")

                if input.shape[1] > max_length:
                    input = input[:, :max_length]
                preds, _ = model(input)
                preds = preds.argmax(dim=1)

                tot += float(input.size(0))
                cor += float((output == preds).sum().item())

            acc = cor / tot

    print("Epoch:{}; Loss: {}; Accuracy: {}".format(epoch, loss.item(), acc))