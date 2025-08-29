# MIT License
# Copyright (c) 2025
#
# This is part of the dino_tutorial package
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
# For additional questions contact Thomas O'Leary-Roseberry

from copy import deepcopy
import torch
import torch.nn as nn

from tqdm import tqdm

from .metrics import squared_f_norm, squared_f_error


def l2_training(model,loss_func,train_loader, validation_loader,\
                     optimizer,lr_scheduler=None,n_epochs = 100, verbose = False):
    device = next(model.parameters()).device

    train_history = {}
    train_history['train_loss_l2'] = []
    train_history['validation_loss_l2'] = []

    for epoch in range(n_epochs):
        # Training
        train_loss = 0
        model.train()
        for batch in train_loader:
            m, u = batch
            m = m.to(device)
            u = u.to(device)
            u_pred = model(m)
            loss = loss_func(u_pred, u)
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            train_loss += loss.item() * m.shape[0]

        train_loss /= len(train_loader.dataset)
        
        train_history['train_loss_l2'].append(train_loss)

        # Evaluation
        with torch.no_grad():
            model.eval()
            validation_loss = 0
            for batch in validation_loader:
                m, u = batch
                m = m.to(device)
                u = u.to(device)
                u_pred = model(m)
                loss = loss_func(u_pred, u)
                validation_loss += loss.item() * m.shape[0]
        validation_loss /= len(validation_loader.dataset)

        # Update learning rate if lr_scheduler is provided
        if lr_scheduler is not None:
            lr_scheduler.step(validation_loss)
        if epoch %20 == 0 and verbose:
            print('epoch = ',epoch)
            print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.6e}")
            print(f"Epoch {epoch+1}/{n_epochs}, Validation Loss: {validation_loss:.6e}")

    return model, train_history

def h1_training(model,loss_func_l2,loss_func_jac,train_loader, validation_loader,\
                     optimizer,lr_scheduler=None,n_epochs = 100, verbose = False,\
                     mode="forward", jac_weight = 1.0, output_projector = None,\
                     slow_jac = False):
    device = next(model.parameters()).device

    if output_projector is None:
        def forward_pass(m):
            return model(torch.reshape(m, (-1, m.shape[-1])))
    else:
        output_projector = output_projector.to(device)
        # This case assumes a full state output and reduces it pre-emptively.
        def forward_pass(m):
            return model(torch.reshape(m, (-1, m.shape[-1])))@ output_projector

    if mode == "forward":
        jac_func = torch.func.vmap(torch.func.jacfwd(forward_pass))
    elif mode == "reverse":
        jac_func = torch.func.vmap(torch.func.jacrev(forward_pass))
    else:
        raise ValueError("Jacobian mode must be either 'forward' or 'reverse'")

    train_history = {}
    train_history['train_loss_l2'] = []
    train_history['validation_loss'] = []
    train_history['validation_loss_l2'] = []
    train_history['validation_loss_jac'] = []

    for epoch in range(n_epochs):
        # Training
        train_loss = 0
        model.train()
        for batch in train_loader:
            m, u, J = batch
            m = m.to(device)
            u = u.to(device)
            J = J.to(device)
            u_pred = model(m)
            if slow_jac:
                J_pred = [jac_func(m[i:i+1]) for i in range(m.shape[0])]
                J_pred = torch.cat(J_pred, dim=0)
            else:
                J_pred = jac_func(m)
            # J_pred = jac_func(m)
            loss_l2 = loss_func_l2(u_pred, u)
            loss_jac = loss_func_jac(J_pred, J)
            loss = loss_l2 + jac_weight * loss_jac
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * m.shape[0]


        train_loss /= len(train_loader.dataset)
        
        train_history['train_loss_l2'].append(train_loss)

        # Evaluation
        with torch.no_grad():
            model.eval()
            validation_loss = 0
            validation_loss_l2 = 0
            validation_loss_jac = 0
            for batch in validation_loader:
                m, u, J = batch
                m = m.to(device)
                u = u.to(device)
                J = J.to(device)
                u_pred = model(m)
                loss_l2 = loss_func_l2(u_pred, u)
                loss_jac = loss_func_jac(jac_func(m), J)
                loss = loss_l2 + jac_weight * loss_jac
                validation_loss += loss.item() * m.shape[0]
                validation_loss_l2 += loss_l2.item() * m.shape[0]
                validation_loss_jac += loss_jac.item() * m.shape[0]

        validation_loss /= len(validation_loader.dataset) 
        validation_loss_l2 /=len(validation_loader.dataset) 
        validation_loss_jac /= len(validation_loader.dataset) 

        # Update learning rate if lr_scheduler is provided
        if lr_scheduler is not None:
            lr_scheduler.step(validation_loss)

        train_history['validation_loss'].append(validation_loss)
        train_history['validation_loss_l2'].append(validation_loss_l2)
        train_history['validation_loss_jac'].append(validation_loss_jac)

        # # Evaluation
        # with torch.no_grad():
        #     model.eval()
        #     validation_loss = 0
        #     for batch in validation_loader:
        #         m, u, J = batch
        #         m = m.to(device)
        #         u = u.to(device)
        #         u_pred = model(m)
        #         loss = loss_func(u_pred, u)
        #         validation_loss += loss.item() * m.shape[0]
        # validation_loss /= len(validation_loader.dataset)

        # # Update learning rate if lr_scheduler is provided
        # if lr_scheduler is not None:
        #     lr_scheduler.step(validation_loss)
        if epoch %10 == 0 and verbose:
            print('epoch = ',epoch)
            print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.6e}")
            print(f"Epoch {epoch+1}/{n_epochs}, Validation Loss: {validation_loss:.6e}")
            print(f"Epoch {epoch+1}/{n_epochs}, Validation Loss L2: {validation_loss_l2:.6e}")
            print(f"Epoch {epoch+1}/{n_epochs}, Validation Loss Jac: {validation_loss_jac:.6e}")

    return model, train_history



def h1_training_fno(model,
                    loss_func_l2,
                    loss_func_jac,
                    train_loader,
                    validation_loader,
                    optimizer,
                    input_basis = None,
                    output_projector = None,
                    lr_scheduler=None,
                    n_epochs=100,
                    verbose=False,
                    mode="reverse",
                    jac_weight=1.0,
                    num_splits=4):
    device = next(model.parameters()).device

    if mode == "forward":
        assert input_basis is not None
        raise
        def fwd_sensitivity(m, basis):
            _, grad = torch.func.jvp(model, (m.reshape(1, -1),), (basis.reshape(1, -1),))
            return grad.reshape(-1)
        def fwd_sensitivity_batched(m, start_idx, end_idx):
            grads = torch.vmap(fwd_sensitivity, in_dims=(None, 1), out_dims=0)(m, input_basis[:, start_idx:end_idx])
            return grads
        jac_func = torch.vmap(fwd_sensitivity_batched, in_dims=(0, None, None), out_dims=0)
    elif mode == "reverse":
        assert output_projector is not None
        output_projector = output_projector.to(device)
        def backward_sensitivity(m, start_idx, end_idx):
            results = model(torch.reshape(m, (-1, m.shape[-1]))) @ output_projector[:, start_idx:end_idx]
            return results.flatten()
        jac_func = torch.vmap(torch.func.jacrev(backward_sensitivity), in_dims=(0, None, None), out_dims=0)
    else:
        raise ValueError("Jacobian mode must be either 'forward' or 'reverse'")

    train_history = {}
    train_history['train_loss_l2'] = []
    train_history['train_loss_jac'] = []
    train_history['validation_loss'] = []
    train_history['validation_loss_l2'] = []
    train_history['validation_loss_jac'] = []
    best_loss = float('inf')

    for epoch in range(n_epochs):
        # Training
        train_loss_l2 = 0
        train_loss_jac = 0
        model.train()
        for batch in tqdm(train_loader):
            # Initialize accumulated gradients
            accumulated_grads = [torch.zeros_like(param) for param in model.parameters()]
            m, u, J = batch
            m = m.to(device)
            u = u.to(device)
            J = J.to(device)

            optimizer.zero_grad()
            u_pred = model(m)
            loss_l2 = loss_func_l2(u_pred, u)
            loss_l2.backward()
            train_loss_l2 += loss_l2.item() * m.shape[0]

            # Retrieve gradients from l2 loss computation
            for param, acc_grad in zip(model.parameters(), accumulated_grads):
                if param.grad is not None:
                    acc_grad += param.grad.detach()

            normalizations = torch.vmap(squared_f_norm)(J)
            if mode == "forward":
                shard_size = (input_basis.shape[-1] + num_splits - 1) // num_splits

                for i in range(num_splits):
                    if i != num_splits - 1:
                        start_idx = i * shard_size
                        end_idx = (i + 1) * shard_size
                    else:
                        start_idx = i * shard_size
                        end_idx = input_basis.shape[-1]

                    optimizer.zero_grad() # Clear the gradients for the new shard
                    J_pred = jac_func(m, start_idx, end_idx)
                    J_pred = J_pred @ output_projector
                    J_pred = J_pred.transpose(1, 2)

                    loss_jac = torch.mean(torch.vmap(squared_f_error, in_dims=(0, 0), out_dims=0)(J_pred, J[:, :, start_idx:end_idx]) / normalizations)
                    loss_jac.backward()
                    train_loss_jac += loss_jac.item() * m.shape[0]

            elif mode == "reverse":
                shard_size = (output_projector.shape[-1] + num_splits - 1) // num_splits

                for i in range(num_splits):
                    if i != num_splits - 1:
                        start_idx = i * shard_size
                        end_idx = (i + 1) * shard_size
                    else:
                        start_idx = i * shard_size
                        end_idx = output_projector.shape[-1]

                    optimizer.zero_grad()
                    J_pred = jac_func(m, start_idx, end_idx)
                    J_pred = J_pred @ input_basis

                    loss_jac = torch.mean(torch.vmap(squared_f_error, in_dims=(0, 0), out_dims=0)(J_pred, J[:, start_idx:end_idx, :]) / normalizations)
                    loss_jac.backward()
                    train_loss_jac += loss_jac.item() * m.shape[0]

            # Retrieve gradients from the jacobian loss on the shard
            for param, acc_grad in zip(model.parameters(), accumulated_grads):
                if param.grad is not None:
                    acc_grad += jac_weight * param.grad.detach()

            # Manually set the gradients
            for param, acc_grad in zip(model.parameters(), accumulated_grads):
                param.grad = acc_grad

            optimizer.step()

        train_loss_l2 /= len(train_loader.dataset)
        train_loss_jac /= len(train_loader.dataset)

        train_history['train_loss_l2'].append(train_loss_l2)
        train_history['train_loss_jac'].append(train_loss_jac)

        # Evaluation
        with torch.no_grad():
            model.eval()
            validation_loss_l2 = 0
            validation_loss_jac = 0
            for batch in validation_loader:
                m, u, J = batch
                m = m.to(device)
                u = u.to(device)
                J = J.to(device)
                u_pred = model(m)
                loss_l2 = loss_func_l2(u_pred, u)
                validation_loss_l2 += loss_l2.item() * m.shape[0]

                normalizations = torch.vmap(squared_f_norm)(J)
                if mode == "forward":
                    shard_size = (input_basis.shape[-1] + num_splits - 1) // num_splits

                    for i in range(num_splits):
                        if i != num_splits - 1:
                            start_idx = i * shard_size
                            end_idx = (i + 1) * shard_size
                        else:
                            start_idx = i * shard_size
                            end_idx = input_basis.shape[-1]

                        J_pred = jac_func(m, start_idx, end_idx)
                        J_pred = output_projector.T @ J_pred.T

                        loss_jac = torch.mean(torch.vmap(squared_f_error, in_dims=(0, 0), out_dims=0)(J_pred, J[:, :, start_idx:end_idx]) / normalizations)
                        validation_loss_jac += loss_jac.item() * m.shape[0]

                elif mode == "reverse":
                    shard_size = (output_projector.shape[-1] + num_splits - 1) // num_splits
                    for i in range(num_splits):
                        if i != num_splits - 1:
                            start_idx = i * shard_size
                            end_idx = (i + 1) * shard_size
                        else:
                            start_idx = i * shard_size
                            end_idx = output_projector.shape[-1]

                        J_pred = jac_func(m, start_idx, end_idx)
                        J_pred = J_pred @ input_basis

                        loss_jac = torch.mean(torch.vmap(squared_f_error, in_dims=(0, 0), out_dims=0)(J_pred, J[:, start_idx:end_idx, :]) / normalizations)
                        validation_loss_jac += loss_jac.item() * m.shape[0]


        validation_loss = validation_loss_l2 + validation_loss_jac * jac_weight
        validation_loss /= len(validation_loader.dataset)
        validation_loss_l2 /=len(validation_loader.dataset)
        validation_loss_jac /= len(validation_loader.dataset)
        if best_loss > validation_loss:
            best_loss = validation_loss
            best_model = deepcopy(model)

        # Update learning rate if lr_scheduler is provided
        if lr_scheduler is not None:
            if 'metrics' in lr_scheduler.step.__code__.co_varnames:
                lr_scheduler.step(validation_loss)
            else:
                lr_scheduler.step()
            print(f"Learning rate: {optimizer.param_groups[0]['lr']}")

        train_history['validation_loss'].append(validation_loss)
        train_history['validation_loss_l2'].append(validation_loss_l2)
        train_history['validation_loss_jac'].append(validation_loss_jac)

        if verbose:
            print('epoch = ',epoch)
            print(f"Epoch {epoch+1}/{n_epochs}, Train Loss L2: {train_loss_l2:.6e}")
            print(f"Epoch {epoch+1}/{n_epochs}, Train Loss Jac: {train_loss_jac:.6e}")
            print(f"Epoch {epoch+1}/{n_epochs}, Validation Loss L2: {validation_loss_l2:.6e}")
            print(f"Epoch {epoch+1}/{n_epochs}, Validation Loss Jac: {validation_loss_jac:.6e}")

    return best_model, train_history
