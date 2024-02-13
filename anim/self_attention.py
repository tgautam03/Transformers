import numpy as np
from manim import *

class SelfAttention(Scene):
    def construct(self):
        self.camera.background_color = "#282a36"
        title = Text("Self Attention")
        box = Rectangle(width=16/9*2.25, height=2.25, color=WHITE)
        self.play(Write(title))
        self.wait(1)
        self.play(Create(box))
        self.wait(1)
        in_seq = Matrix([[1, 2, 3, 4, 5]]).scale(0.7).next_to(box, LEFT)
        out_seq = Matrix([[5, 4, 3, 2, 1]]).scale(0.7).next_to(box, RIGHT)
        self.play(Write(in_seq))
        self.wait(1)
        self.play(FadeOut(in_seq, target_position=title))
        self.play(FadeIn(out_seq, target_position=title))
        self.wait(1)
        self.play(FadeOut(title, out_seq))
        self.play(box.animate.scale(4), run_time=2)
        self.wait(1)