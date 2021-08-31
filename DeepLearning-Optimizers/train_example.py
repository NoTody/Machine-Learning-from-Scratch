# example of using written optimizer for training
import torch
import Adagrad.Adagrad, RMSprop.RMSprop, Adam.Adam
from tqdm import tqdm

# train function
def train_model(model, criterion, optimizer, train_loader, val_loader, epochs, device, tol):
    # set objects for storing metrics
    tr_losses = []
    val_losses = []
    tr_accs = []
    val_accs = []
    # track history of validation loss to perform early stopping
    # set number of epochs to track
    tol = tol
    # initialize states for optimizer
    states = init_states(model.parameters())
    # Train model
    for epoch in range(epochs):
        # training
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        print(f'Epoch {epoch}:')
        print('Train:')
        tr_loss, nb_tr_steps, correct, tot_correct = 0, 0, 0, 0
        model.train()
        for batch_idx, (data, target) in loop:
            data, target = data.to(device), target.to(device)
            #optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            # optimizer "step"
            if optimizer == 'Adagrad':
                Adagrad(model.parameters(), states, {'lr':0.01, 'eps':1e-10})
            elif optimizer == 'RMSprop':
                RMSprop(model.parameters(), states, {'lr':0.001, 'eps':1e-8, 'gamma':0.99})
            elif optimizer == 'Adam':
                Adam(model.parameters(), states, {'lr':0.001, 'eps':1e-8, 'betas':(0.9, 0.999)})
            else:
                print('Wrong optimizer input.')
                break
            #optimizer.step()
            # accumulate loss/step in epoch
            tr_loss += loss.item()
            # calculate accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct = (pred.flatten() == target).sum().item()
            tot_correct += correct
            nb_tr_steps += 1
            # update progress bar
            loop.set_description(f'Epoch [{epoch}/{epochs}]')
            loop.set_postfix(loss = loss.item(), acc = correct / len(target))

        tr_acc = tot_correct / len(train_loader.dataset)
        tr_loss = tr_loss / nb_tr_steps
        tr_losses.append(tr_loss)
        tr_accs.append(tr_acc)
        print(f'Train Accuracy: {tr_acc}')
        print(f'Train Loss: {tr_loss}')
        
        # validation
        model.eval()
        val_loss, nb_val_steps, correct, tot_correct = 0, 0, 0, 0
        print('Validation:')
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                # sum up batch loss
                loss = criterion(output, target)
                # get the index of the max log-probability
                pred = output.argmax(dim=1)
                correct = (pred == target).sum().item()
                tot_correct += correct
                val_loss += loss.item()
                nb_val_steps += 1
        val_acc = tot_correct / len(val_loader.dataset)
        val_loss = val_loss / nb_val_steps
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        print(f'Validation Accuracy: {val_acc}')
        print(f'Validation Loss: {val_loss}')
        # check validation loss history for early stopping
        if len(val_losses) > tol:
            losses_diff_hist = []
            tracked_loss = val_losses[len(val_losses)-tol-1]
            # get last 'tol' tolerance index and calculate loss difference history
            for i in range(1, tol+1):
                losses_diff_hist.append(val_losses[len(val_losses)-i] - tracked_loss)
            print(losses_diff_hist)
            # if all histories are larger than or equal previous tracked loss, stop training
            # larger than 0 means the losses are not decreasing
            if sum([loss_diff >= 0 for loss_diff in losses_diff_hist]) == tol:
                print(sum([loss_diff >= 0 for loss_diff in losses_diff_hist]))
                break
    return tr_accs, val_accs, tr_losses, val_losses