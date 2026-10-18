import pytest
from streamlit.testing.v1 import AppTest

def test_dashboard_renders():
    # Test that the dashboard renders without crashing
    at = AppTest.from_file("../src/ui/dashboard.py", default_timeout=10)
    
    
    # Run the app
    at.run()
    
    # Verify the tabs are generated (Streamlit places tabs in layout structures)
    assert len(at.tabs) == 4
    
    # Test Tab 1: Live Simulator state
    assert len(at.header) > 0
    assert "Live Interactive Simulator" in at.header[0].value

def test_dashboard_interactive_logic():
    at = AppTest.from_file("../src/ui/dashboard.py", default_timeout=10)
    at.run()
    
    # Text input 0 should be VM Name
    at.text_input[0].set_value("TestNode_1")
    # Number input 0 should be CPU Capacity (in the VM form)
    at.number_input[0].set_value(2000.0) 
    
    # Find the "Provision VM" button and click it
    prov_btn = next((b for b in at.button if "Provision VM" in b.label), None)
    if prov_btn:
        prov_btn.click().run()
        # Verify it updated state
        assert "TestNode_1" in at.session_state.env.vms
        assert len(at.session_state.history) == 1
    
    # Click Create Workload
    # WL CPU is the 4th number input in the file
    at.number_input[4].set_value(500.0)
    queue_btn = next((b for b in at.button if "Queue Workload" in b.label), None)
    if queue_btn:
        queue_btn.click().run()
        # Verify it updated queue
        assert len(at.session_state.queue) == 1
    
    # Step simulation
    step_btn = next((b for b in at.button if "Step" in b.label), None)
    if step_btn and prov_btn and queue_btn:
        step_btn.click().run()
        
        # Workload should be out of queue and into VM if it fits
        assert len(at.session_state.queue) == 0
        assert at.session_state.env.vms["TestNode_1"].cpu_alloc >= 500.0
