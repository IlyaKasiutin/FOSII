import numpy as np
from typing import Optional, Dict

from .module import Module
from .layers.gru import GRU as GRUBlock
import tqdm


class SequenceGRU(Module):
    def __init__(self, n_inputs: int, n_hidden: int, n_outputs: int,
                 time_steps: int = 5):
        """
        Simple GRU model that unrolls the given GRU block for a fixed number
        of steps (default 5). Follows project patterns from cnn.py, mlp.py, and rnn_model.py.

        Note: The provided GRU block uses matrices with the same
        second dimension. To keep shapes consistent with that implementation,
        it is recommended to set n_inputs == n_hidden == n_outputs.
        """
        super().__init__()
        self.time_steps = time_steps
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs

        # Use the provided single-step GRU block
        # The block internally maintains its state via prev_h
        self.block = GRUBlock(n_inputs=n_inputs, n_hidden=n_hidden, n_outputs=n_outputs)

        # For a simple many-to-one setup, we interpret the block's last output
        # as logits for classification/regression sized to n_outputs.
        # Given the block shapes, best compatibility is when n_hidden == n_outputs.

        self._modules = {"gru_block": self.block}

        # Caches for BPTT
        self._cache: Dict[str, list] = {
            "xs": [],        # list of (n_in,) or (n_in, batch)
            "prev_hs": [],   # list of states before each step
            "ys": []         # list of outputs (n_out,) or (n_out, batch)
        }

    def _reset_state(self) -> None:
        # Reset the block's recurrent state to zeros for a new sequence
        # Match the existing storage in the block
        self.block.prev_h = np.zeros(self.n_hidden)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass over a sequence for many-to-one prediction.

        Args:
            x: Input of shape (time_steps, n_inputs)

        Returns:
            y: Output logits of shape (n_outputs,) from the last step
        """
        # Clear caches
        self._cache["xs"].clear()
        self._cache["prev_hs"].clear()
        self._cache["ys"].clear()

        # Reset state per sequence
        self._reset_state()

        y_last = None
        for t in range(self.time_steps):
            # Prepare step input
            x_t = x[t, :]

            # Cache state before step for BPTT
            prev_h_t = self.block.prev_h.copy()

            y_t = self.block.single_step(x_t)  # (n_out,)

            # Update caches
            self._cache["xs"].append(x_t)
            self._cache["prev_hs"].append(prev_h_t)
            self._cache["ys"].append(y_t)

            # For next step, the block keeps its internal state
            y_last = y_t

        # Return last step output
        return y_last

    def backward(self, loss_grad: np.ndarray) -> None:
        """
        Backward-through-time for many-to-one loss.

        Args:
            loss_grad: dL/dy_last of shape (n_outputs,)
        """
        # Initialize accumulated grads for the block's parameters
        dW_r_total = np.zeros_like(self.block.W_r)
        dU_r_total = np.zeros_like(self.block.U_r)
        dW_z_total = np.zeros_like(self.block.W_z)
        dU_z_total = np.zeros_like(self.block.U_z)
        dW_h_total = np.zeros_like(self.block.W_h)
        dU_h_total = np.zeros_like(self.block.U_h)
        dV_total = np.zeros_like(self.block.V)

        # diff_h is gradient flowing into hidden state from future time steps
        diff_h = np.zeros(self.n_hidden)

        # Only the last step receives direct gradient from the loss in many-to-one
        for t in reversed(range(self.time_steps)):
            x_t = self._cache["xs"][t]
            prev_h_t = self._cache["prev_hs"][t]

            if t == self.time_steps - 1:
                # loss_grad is (n_out,)
                dmulv = loss_grad
            else:
                # No direct loss on earlier outputs in many-to-one
                dmulv = np.zeros_like(self._cache["ys"][t])

            dprev_h, dW_r, dU_r, dW_z, dU_z, dW_h, dU_h, dV = self.block.backward(
                x_t, prev_h_t, diff_h, dmulv
            )

            dW_r_total += dW_r
            dU_r_total += dU_r
            dW_z_total += dW_z
            dU_z_total += dU_z
            dW_h_total += dW_h
            dU_h_total += dU_h
            dV_total += dV

            # Propagate state gradient to previous time step
            diff_h = dprev_h

        # Stash grads on the block
        self.block.W_r_grad = dW_r_total
        self.block.U_r_grad = dU_r_total
        self.block.W_z_grad = dW_z_total
        self.block.U_z_grad = dU_z_total
        self.block.W_h_grad = dW_h_total
        self.block.U_h_grad = dU_h_total
        self.block.V_grad = dV_total

    def update_params(self, learning_rate: float = 0.001, optimizer=None) -> None:
        if optimizer is not None:
            optimizer.step(self)
            return

        # Apply simple SGD using accumulated gradients
        if hasattr(self.block, "W_r_grad"):
            self.block.W_r -= learning_rate * self.block.W_r_grad
        if hasattr(self.block, "U_r_grad"):
            self.block.U_r -= learning_rate * self.block.U_r_grad
        if hasattr(self.block, "W_z_grad"):
            self.block.W_z -= learning_rate * self.block.W_z_grad
        if hasattr(self.block, "U_z_grad"):
            self.block.U_z -= learning_rate * self.block.U_z_grad
        if hasattr(self.block, "W_h_grad"):
            self.block.W_h -= learning_rate * self.block.W_h_grad
        if hasattr(self.block, "U_h_grad"):
            self.block.U_h -= learning_rate * self.block.U_h_grad
        if hasattr(self.block, "V_grad"):
            self.block.V -= learning_rate * self.block.V_grad

    def zero_grad(self) -> None:
        if hasattr(self.block, "W_r_grad"):
            self.block.W_r_grad.fill(0)
        if hasattr(self.block, "U_r_grad"):
            self.block.U_r_grad.fill(0)
        if hasattr(self.block, "W_z_grad"):
            self.block.W_z_grad.fill(0)
        if hasattr(self.block, "U_z_grad"):
            self.block.U_z_grad.fill(0)
        if hasattr(self.block, "W_h_grad"):
            self.block.W_h_grad.fill(0)
        if hasattr(self.block, "U_h_grad"):
            self.block.U_h_grad.fill(0)
        if hasattr(self.block, "V_grad"):
            self.block.V_grad.fill(0)

    def train_step(self, X: np.ndarray, y_true: np.ndarray, loss_fn,
                   learning_rate: float = 0.001, optimizer=None) -> float:
        """
        Run a single optimization step on a sequence.

        Args:
            X: (time_steps, n_inputs)
            y_true: (n_outputs,) one-hot for CE or targets for regression
        """
        y_pred = self.forward(X)  # (n_outputs,)

        # Align with patterns in cnn/mlp losses
        loss_vals = loss_fn.forward(y_pred.T, y_true.T)
        batch_loss = float(np.mean(loss_vals))

        loss_grad = loss_fn.backward(y_pred.T, y_true.T).T  # (n_outputs,)
        self.backward(loss_grad)

        self.update_params(learning_rate, optimizer)
        self.zero_grad()

        return batch_loss

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              epochs: int = 10, batch_size: int = 32, learning_rate: float = 0.001,
              loss_fn=None, optimizer=None, verbose: bool = True) -> dict:
        if loss_fn is None:
            # Defer import to avoid circulars if any
            from .losses.cross_entropy import CrossEntropy
            loss_fn = CrossEntropy()

        num_samples = X_train.shape[0]
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }

        for epoch in range(epochs):
            indices = np.random.permutation(num_samples)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            total_loss = 0.0
            num_batches = 0

            for start in tqdm.tqdm(range(0, num_samples, batch_size), total=(num_samples / batch_size)):
                end = start + batch_size
                X_batch = X_train_shuffled[start:end]
                y_batch = y_train_shuffled[start:end]

                for elem in zip(X_batch, y_batch):
                    cur_x = elem[0]
                    cur_y = elem[1]
                    batch_loss = self.train_step(
                        cur_x, cur_y, loss_fn, learning_rate, optimizer
                    )
                    total_loss += batch_loss
                    num_batches += 1

            avg_train_loss = total_loss / num_samples
            history['train_loss'].append(avg_train_loss)

            val_loss = None
            val_accuracy = None
            if X_val is not None and y_val is not None:
                y_val_pred = self.forward(X_val)
                val_loss_vals = loss_fn.forward(y_val_pred.T, y_val.T)
                val_loss = float(np.mean(val_loss_vals))
                history['val_loss'].append(val_loss)

                # If y_val is one-hot/probabilities, compute accuracy over classes
                try:
                    y_val_pred_classes = np.argmax(y_val_pred, axis=1)
                    y_val_true_classes = np.argmax(y_val, axis=1)
                    val_accuracy = float(np.mean(y_val_pred_classes == y_val_true_classes))
                    history['val_accuracy'].append(val_accuracy)
                except Exception:
                    pass

            if verbose:
                if val_loss is not None and val_accuracy is not None:
                    print(
                        f"Epoch {epoch+1}/{epochs} - loss: {avg_train_loss:.4f} "
                        f"- val_loss: {val_loss:.4f} - val_accuracy: {val_accuracy:.4f}"
                    )
                elif val_loss is not None:
                    print(
                        f"Epoch {epoch+1}/{epochs} - loss: {avg_train_loss:.4f} "
                        f"- val_loss: {val_loss:.4f}"
                    )
                else:
                    print(f"Epoch {epoch+1}/{epochs} - loss: {avg_train_loss:.4f}")

        return history

