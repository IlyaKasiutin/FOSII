import numpy as np
from typing import Optional, Dict

from .module import Module
from .layers.lstm import LSTM as LSTMBlock
import tqdm


class SequenceLSTM(Module):
    def __init__(self, n_inputs: int, n_hidden: int, n_outputs: int,
                 time_steps: int = 5):
        super().__init__()
        self.time_steps = time_steps
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs

        self.block = LSTMBlock(n_inputs=n_inputs, n_hidden=n_hidden, n_outputs=n_outputs)

        self._modules = {"lstm_block": self.block}

        # Caches for BPTT
        self._cache: Dict[str, list] = {
            "xs": [],        # list of (n_in,) or (n_in, batch)
            "prev_hs": [],   # list of hidden states before each step
            "prev_cs": [],   # list of cell states before each step
            "ys": []         # list of outputs (n_out,) or (n_out, batch)
        }

    def _reset_state(self) -> None:
        self.block.prev_h = np.zeros(self.n_hidden)
        self.block.prev_c = np.zeros(self.n_hidden)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass over a sequence for many-to-one prediction.

        Args:
            x: Input of shape (time_steps, n_inputs)

        Returns:
            y: Output logits of shape (n_outputs,) from the last step
        """
        self._cache["xs"].clear()
        self._cache["prev_hs"].clear()
        self._cache["prev_cs"].clear()
        self._cache["ys"].clear()

        self._reset_state()

        y_last = None
        for t in range(self.time_steps):
            x_t = x[t, :]

            prev_h_t = self.block.prev_h.copy()
            prev_c_t = self.block.prev_c.copy()

            y_t = self.block.single_step(x_t)  # (n_out,)

            self._cache["xs"].append(x_t)
            self._cache["prev_hs"].append(prev_h_t)
            self._cache["prev_cs"].append(prev_c_t)
            self._cache["ys"].append(y_t)

            y_last = y_t

        return y_last

    def backward(self, loss_grad: np.ndarray) -> None:
        """
        Backward-through-time for many-to-one loss.

        Args:
            loss_grad: dL/dy_last of shape (n_outputs,)
        """
        dW_f_total = np.zeros_like(self.block.W_f)
        dU_f_total = np.zeros_like(self.block.U_f)
        db_f_total = np.zeros_like(self.block.b_f)
        
        dW_i_total = np.zeros_like(self.block.W_i)
        dU_i_total = np.zeros_like(self.block.U_i)
        db_i_total = np.zeros_like(self.block.b_i)
        
        dW_c_total = np.zeros_like(self.block.W_c)
        dU_c_total = np.zeros_like(self.block.U_c)
        db_c_total = np.zeros_like(self.block.b_c)
        
        dW_o_total = np.zeros_like(self.block.W_o)
        dU_o_total = np.zeros_like(self.block.U_o)
        db_o_total = np.zeros_like(self.block.b_o)
        
        dV_total = np.zeros_like(self.block.V)

        diff_h = np.zeros(self.n_hidden)
        diff_c = np.zeros(self.n_hidden)

        # Only the last step receives direct gradient from the loss in many-to-one
        for t in reversed(range(self.time_steps)):
            x_t = self._cache["xs"][t]
            prev_h_t = self._cache["prev_hs"][t]
            prev_c_t = self._cache["prev_cs"][t]

            if t == self.time_steps - 1:
                # loss_grad is (n_out,)
                dmulv = loss_grad
            else:
                # No direct loss on earlier outputs in many-to-one
                dmulv = np.zeros_like(self._cache["ys"][t])

            dprev_h, dprev_c, dW_f, dU_f, db_f, dW_i, dU_i, db_i, \
            dW_c, dU_c, db_c, dW_o, dU_o, db_o, dV = self.block.backward(
                x_t, prev_h_t, prev_c_t, diff_h, diff_c, dmulv
            )

            dW_f_total += dW_f
            dU_f_total += dU_f
            db_f_total += db_f
            
            dW_i_total += dW_i
            dU_i_total += dU_i
            db_i_total += db_i
            
            dW_c_total += dW_c
            dU_c_total += dU_c
            db_c_total += db_c
            
            dW_o_total += dW_o
            dU_o_total += dU_o
            db_o_total += db_o
            
            dV_total += dV

            diff_h = dprev_h
            diff_c = dprev_c

        self.block.W_f_grad = dW_f_total
        self.block.U_f_grad = dU_f_total
        self.block.b_f_grad = db_f_total
        
        self.block.W_i_grad = dW_i_total
        self.block.U_i_grad = dU_i_total
        self.block.b_i_grad = db_i_total
        
        self.block.W_c_grad = dW_c_total
        self.block.U_c_grad = dU_c_total
        self.block.b_c_grad = db_c_total
        
        self.block.W_o_grad = dW_o_total
        self.block.U_o_grad = dU_o_total
        self.block.b_o_grad = db_o_total
        
        self.block.V_grad = dV_total

    def update_params(self, learning_rate: float = 0.001, optimizer=None) -> None:
        if optimizer is not None:
            optimizer.step(self)
            return

        if hasattr(self.block, "W_f_grad"):
            self.block.W_f -= learning_rate * self.block.W_f_grad
        if hasattr(self.block, "U_f_grad"):
            self.block.U_f -= learning_rate * self.block.U_f_grad
        if hasattr(self.block, "b_f_grad"):
            self.block.b_f -= learning_rate * self.block.b_f_grad
            
        if hasattr(self.block, "W_i_grad"):
            self.block.W_i -= learning_rate * self.block.W_i_grad
        if hasattr(self.block, "U_i_grad"):
            self.block.U_i -= learning_rate * self.block.U_i_grad
        if hasattr(self.block, "b_i_grad"):
            self.block.b_i -= learning_rate * self.block.b_i_grad
            
        if hasattr(self.block, "W_c_grad"):
            self.block.W_c -= learning_rate * self.block.W_c_grad
        if hasattr(self.block, "U_c_grad"):
            self.block.U_c -= learning_rate * self.block.U_c_grad
        if hasattr(self.block, "b_c_grad"):
            self.block.b_c -= learning_rate * self.block.b_c_grad
            
        if hasattr(self.block, "W_o_grad"):
            self.block.W_o -= learning_rate * self.block.W_o_grad
        if hasattr(self.block, "U_o_grad"):
            self.block.U_o -= learning_rate * self.block.U_o_grad
        if hasattr(self.block, "b_o_grad"):
            self.block.b_o -= learning_rate * self.block.b_o_grad
            
        if hasattr(self.block, "V_grad"):
            self.block.V -= learning_rate * self.block.V_grad

    def zero_grad(self) -> None:
        if hasattr(self.block, "W_f_grad"):
            self.block.W_f_grad.fill(0)
        if hasattr(self.block, "U_f_grad"):
            self.block.U_f_grad.fill(0)
        if hasattr(self.block, "b_f_grad"):
            self.block.b_f_grad.fill(0)
            
        if hasattr(self.block, "W_i_grad"):
            self.block.W_i_grad.fill(0)
        if hasattr(self.block, "U_i_grad"):
            self.block.U_i_grad.fill(0)
        if hasattr(self.block, "b_i_grad"):
            self.block.b_i_grad.fill(0)
            
        if hasattr(self.block, "W_c_grad"):
            self.block.W_c_grad.fill(0)
        if hasattr(self.block, "U_c_grad"):
            self.block.U_c_grad.fill(0)
        if hasattr(self.block, "b_c_grad"):
            self.block.b_c_grad.fill(0)
            
        if hasattr(self.block, "W_o_grad"):
            self.block.W_o_grad.fill(0)
        if hasattr(self.block, "U_o_grad"):
            self.block.U_o_grad.fill(0)
        if hasattr(self.block, "b_o_grad"):
            self.block.b_o_grad.fill(0)
            
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
        batch_loss = 0
        for elem in zip(X, y_true):
            cur_x = elem[0]
            cur_y = elem[1]

            y_pred = self.forward(cur_x)  # (n_outputs,)

            loss_vals = loss_fn.forward(y_pred.T, cur_y.T)
            batch_loss += float(np.mean(loss_vals))

            loss_grad = loss_fn.backward(y_pred.T, cur_y.T).T  # (n_outputs,)
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

                batch_loss = self.train_step(
                    X_batch, y_batch, loss_fn, learning_rate, optimizer
                )
                total_loss += batch_loss

            avg_train_loss = total_loss / num_samples
            history['train_loss'].append(avg_train_loss)

            val_loss = 0
            val_accuracy = None
            if X_val is not None and y_val is not None:
                for elem in zip(X_val, y_val):
                    cur_x = elem[0]
                    cur_y = elem[1]
                    y_pred = self.forward(cur_x)  # (1, n_outputs)
                    loss_vals = loss_fn.forward(y_pred.T, cur_y.T)
                    val_loss +=  np.mean(loss_vals)
                val_loss /= X_val.shape[0]
                history['val_loss'].append(val_loss)

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